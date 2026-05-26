"""Print a data-characteristics report for one patient.

Measurement-driven tuning aid. Reads ``<id>_aligned.csv`` and ``<id>_cleaned.csv``;
if ``<id>_final.csv`` exists, also reports per-detector firing rates and the
magnitude histograms of nonzero values.

    python -m pipeline.characterize <patient_id>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import config


def _segments(finite_mask: np.ndarray) -> list[int]:
    """Lengths of contiguous True runs."""
    lengths = []
    n = 0
    for f in finite_mask:
        if f:
            n += 1
        elif n:
            lengths.append(n)
            n = 0
    if n:
        lengths.append(n)
    return lengths


def _hist(x: np.ndarray, bins) -> str:
    h, edges = np.histogram(x, bins=bins)
    parts = [f"  [{edges[i]:>6.3f}, {edges[i+1]:>6.3f}): {h[i]}" for i in range(len(h))]
    return "\n".join(parts)


def characterize(patient_id: str) -> None:
    aligned = pd.read_csv(config.processed_path(patient_id, "aligned"))
    cleaned = pd.read_csv(config.processed_path(patient_id, "cleaned"))

    hrv = pd.to_numeric(aligned[config.HRV_RAW_COL], errors="coerce").to_numpy(dtype=float)
    hrv = hrv[np.isfinite(hrv)]
    n = len(hrv)
    print(f"=== Characteristics: {patient_id}  (n={n} HRV readings, "
          f"{len(aligned)} rows, "
          f"{(pd.to_datetime(aligned['timestamp']).max() - pd.to_datetime(aligned['timestamp']).min()).days} days) ===\n")

    # --- HRV distribution -----------------------------------------------------
    skew = float(stats.skew(hrv))
    kurt = float(stats.kurtosis(hrv))  # excess (Normal=0)
    nt = stats.normaltest(hrv)
    print("HRV distribution:")
    print(f"  mean={hrv.mean():.2f}  std={hrv.std():.2f}  min={hrv.min():.0f}  max={hrv.max():.0f}")
    print(f"  skew={skew:+.3f}  kurtosis(excess)={kurt:+.3f}  normaltest p={nt.pvalue:.2e}")
    shape = ("right-skewed" if skew > 0.5 else "left-skewed" if skew < -0.5 else "~symmetric")
    tails = ("leptokurtic (heavy)" if kurt > 0.5 else "platykurtic (flat/bimodal)" if kurt < -0.5 else "~mesokurtic")
    print(f"  -> {shape}, {tails}  (do NOT log-transform unless right-skewed)\n")

    # --- ceiling / clipping ---------------------------------------------------
    near_max = ((hrv >= hrv.max() - 1.0).sum() / n) * 100
    print(f"Ceiling check: {near_max:.2f}% of values within 1 unit of max ({hrv.max():.0f})\n")

    # --- raw lag-1 autocorrelation (epoch summaries vs beat-to-beat) ----------
    def lag1(x):
        x = x[np.isfinite(x)]
        return float(np.corrcoef(x[:-1], x[1:])[0, 1]) if len(x) > 2 else float("nan")
    ac_raw = lag1(hrv)
    print(f"Lag-1 autocorrelation (raw hrvValue): {ac_raw:+.3f}  "
          f"({'epoch-summary-like' if abs(ac_raw) < 0.2 else 'temporally smooth'})")

    smoothed_path = config.processed_path(patient_id, "smoothed")
    ac_sm = None
    if smoothed_path.exists():
        sm = pd.read_csv(smoothed_path)[config.SMOOTHED_COL].to_numpy(dtype=float)
        ac_sm = lag1(sm)
        print(f"Lag-1 autocorrelation (smoothed_value): {ac_sm:+.3f}")
    print()

    # --- gap stats ------------------------------------------------------------
    md = pd.to_numeric(cleaned[config.MINUTE_DIFF_COL], errors="coerce").to_numpy(dtype=float)
    md = md[np.isfinite(md) & (md > 0)]
    cv = float(md.std() / md.mean()) if md.mean() > 0 else float("nan")
    print("Gap stats (minute_diff between consecutive HRV readings):")
    print(f"  median={np.median(md):.1f}  mean={md.mean():.1f}  CV={cv:.2f}  max={md.max():.0f}")
    print(f"  >60min:  {(md > 60).sum()}    >360min(6h): {(md > 360).sum()}\n")

    # --- segment lengths after chunking + VAE coverage table ------------------
    hrv_clean = pd.to_numeric(cleaned[config.HRV_CLEAN_COL], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(hrv_clean)
    segs = np.array(_segments(finite))
    n_finite = int(finite.sum())
    print(f"Segments after chunking: {len(segs)} segments, "
          f"{n_finite} finite samples (masked={(~finite).sum()})")
    pct = np.percentile(segs, [10, 25, 50, 75, 90, 99])
    print(f"  seg-length percentiles  p10={pct[0]:.0f}  p25={pct[1]:.0f}  "
          f"p50={pct[2]:.0f}  p75={pct[3]:.0f}  p90={pct[4]:.0f}  p99={pct[5]:.0f}  "
          f"max={segs.max()}")
    print("  histogram:")
    candidate_edges = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]
    edges = [e for e in candidate_edges if e <= segs.max()] + [int(segs.max()) + 1]
    h, _ = np.histogram(segs, bins=edges)
    for i in range(len(h)):
        print(f"    len [{edges[i]:>4d}, {edges[i+1]:>4d}): {h[i]} segments")

    print("\nVAE coverage by window size (segments long enough -> fraction of points):")
    print(f"  {'window':>8}  {'n_segs>=W':>10}  {'samples_in':>12}  {'% of finite':>11}")
    for W in (128, 64, 32, 16, 8):
        big = segs[segs >= W]
        covered = int(big.sum())
        print(f"  {W:>8d}  {len(big):>10d}  {covered:>12d}  "
              f"{(covered / n_finite * 100 if n_finite else 0):>10.1f}%")

    # --- per-detector firing rate + magnitude histogram -----------------------
    final_path = config.final_path(patient_id)
    if not final_path.exists():
        print("\n(no final CSV yet — re-run after CPD to get per-detector stats)")
        return
    print("\nPer-detector (from final CSV):")
    fin = pd.read_csv(final_path)
    total = len(fin)
    for name in ("bocpd", "kcp", "hmm", "kalman"):
        ind = fin[f"{name}_indicator"].to_numpy()
        mag = fin[f"{name}_magnitude"].to_numpy()
        nz = mag[mag > 1e-6]
        rate = ind.sum() / total * 100
        knee = float(np.percentile(nz, 90)) if len(nz) else float("nan")
        print(f"  [{name}] firing rate = {rate:5.2f}%  "
              f"({int(ind.sum())}/{total})  "
              f"nonzero magnitudes: n={len(nz)}  "
              f"p50={np.median(nz) if len(nz) else float('nan'):.4f}  "
              f"p90(knee)={knee:.4f}  "
              f"max={(nz.max() if len(nz) else 0):.4f}")
        if len(nz) > 5:
            edges = np.linspace(0, max(nz.max(), 1e-6), 11)
            h, _ = np.histogram(nz, bins=edges)
            bars = "  " + "  ".join(f"[{edges[i]:.2f}-{edges[i+1]:.2f}]:{h[i]}" for i in range(len(h)))
            print(bars)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m pipeline.characterize <patient_id>")
        raise SystemExit(2)
    characterize(sys.argv[1])
