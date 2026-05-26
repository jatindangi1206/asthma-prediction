"""Pick a diverse subset of patients to train the VAE on.

Single-patient VAE training biases the smoother to that patient's regime. This
script profiles every patient (raw HRV file only — fast, no pipeline run) along
the dimensions that matter for the smoother:

    volume      n_readings, days_span
    level       mean, ceiling fraction
    variance    std, CV, skew, excess kurtosis (bimodality)
    cadence     median gap, gap CV, fraction of gaps >1h / >6h

Selection uses **farthest-point sampling** on z-scored features:
  - Start from the patient closest to the population centroid (most "average").
  - Iteratively add the patient that MAXIMISES the minimum Euclidean distance
    to the already-selected set.
  - Stop at k=5.

That gives one near-median anchor plus four extremes that span the spread —
covering low/high missingness, low/high variance, regular/irregular cadence,
unimodal/bimodal shape — rather than five random or all-similar patients.

    python -m pipeline.select_training_set [--k 5]

Outputs:
    data/processed/training_set_profile.csv   (full 109-patient profile table)
    prints the selected ids + their distinguishing features.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import config

MIN_READINGS = 200      # skip patients with too little HRV to contribute windows
MIN_DAYS = 7


def _hrv_dir(patient_dir: Path) -> Path | None:
    for name in ("hrv", "HRV"):
        for child in patient_dir.iterdir():
            if child.is_dir() and child.name.lower() == name.lower():
                return child
    return None


def _read_hrv(patient_dir: Path) -> pd.DataFrame | None:
    sub = _hrv_dir(patient_dir)
    if sub is None:
        return None
    parts = sorted(p for p in sub.glob(config.PART_GLOB) if not p.name.startswith("."))
    if not parts:
        return None
    df = pd.read_csv(parts[0])
    # column names: createdTime / hrvValue (per layout) — also fall back to logDateTime
    tcol = next((c for c in ("createdTime", "logDateTime") if c in df.columns), None)
    vcol = next((c for c in ("hrvValue", "hrv_value", "value") if c in df.columns), None)
    if tcol is None or vcol is None:
        return None
    df = pd.DataFrame({
        "_t": pd.to_datetime(df[tcol], errors="coerce"),
        "_v": pd.to_numeric(df[vcol], errors="coerce"),
    }).dropna().sort_values("_t").reset_index(drop=True)
    return df if not df.empty else None


def profile(patient_id: str, patient_dir: Path) -> dict | None:
    df = _read_hrv(patient_dir)
    if df is None or len(df) < MIN_READINGS:
        return None
    v = df["_v"].to_numpy(dtype=float)
    t = df["_t"]
    gaps_min = t.diff().dt.total_seconds().dropna().to_numpy() / 60.0
    days_span = (t.iloc[-1] - t.iloc[0]).total_seconds() / 86400
    if days_span < MIN_DAYS:
        return None
    return {
        "patient_id": patient_id,
        "n_readings": len(v),
        "days_span": days_span,
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "cv": float(np.std(v) / np.mean(v)) if np.mean(v) > 0 else 0.0,
        "skew": float(stats.skew(v)),
        "kurt": float(stats.kurtosis(v)),                # excess (Normal = 0)
        "ceiling_frac": float(((v >= v.max() - 1.0).sum()) / len(v)),
        "cadence_median": float(np.median(gaps_min)) if len(gaps_min) else float("nan"),
        "cadence_cv": float(np.std(gaps_min) / np.mean(gaps_min))
                       if len(gaps_min) and np.mean(gaps_min) > 0 else float("nan"),
        "frac_gaps_gt_1h": float((gaps_min > 60).mean()) if len(gaps_min) else 0.0,
        "frac_gaps_gt_6h": float((gaps_min > 360).mean()) if len(gaps_min) else 0.0,
    }


def _zscore(M: np.ndarray) -> np.ndarray:
    mu = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0)
    sd[sd == 0] = 1.0
    out = (M - mu) / sd
    return np.nan_to_num(out, nan=0.0)


def farthest_point_select(Z: np.ndarray, k: int) -> list[int]:
    """Index into Z of k diverse rows via farthest-point sampling."""
    n = Z.shape[0]
    # Start from the most-average patient (closest to centroid)
    dist_from_centroid = np.linalg.norm(Z - Z.mean(axis=0), axis=1)
    first = int(np.argmin(dist_from_centroid))
    selected = [first]
    min_d = np.linalg.norm(Z - Z[first], axis=1)
    while len(selected) < k:
        nxt = int(np.argmax(min_d))
        if nxt in selected:
            break
        selected.append(nxt)
        new_d = np.linalg.norm(Z - Z[nxt], axis=1)
        min_d = np.minimum(min_d, new_d)
    return selected


def main(k: int = 5) -> list[str]:
    config.ensure_dirs()
    profiles = []
    skipped = 0
    for patient_dir in sorted(config.RAW_DATA_DIR.iterdir()):
        if not patient_dir.is_dir() or patient_dir.name.startswith("."):
            continue
        if patient_dir.name == ".gitkeep":
            continue
        p = profile(patient_dir.name, patient_dir)
        if p is None:
            skipped += 1
            continue
        profiles.append(p)
    if not profiles:
        raise SystemExit("no patients qualified")

    df = pd.DataFrame(profiles).sort_values("patient_id").reset_index(drop=True)
    out_csv = config.PROCESSED_DIR / "training_set_profile.csv"
    df.to_csv(out_csv, index=False)
    print(f"profiled {len(df)} patients (skipped {skipped} below minimums) -> {out_csv}\n")

    feature_cols = ["n_readings", "days_span", "mean", "std", "cv",
                    "skew", "kurt", "ceiling_frac",
                    "cadence_median", "cadence_cv",
                    "frac_gaps_gt_1h", "frac_gaps_gt_6h"]
    # n_readings is log-scaled (heavy right tail across patients)
    feats = df[feature_cols].to_numpy(dtype=float).copy()
    feats[:, 0] = np.log1p(feats[:, 0])
    Z = _zscore(feats)

    chosen_idx = farthest_point_select(Z, k)
    chosen = df.iloc[chosen_idx].reset_index(drop=True)

    print(f"=== Selected {len(chosen)} patients (farthest-point sampling) ===\n")
    # Show distinguishing features for each pick
    pop_mean = df[feature_cols].mean()
    pop_std = df[feature_cols].std(ddof=0).replace(0, 1)
    for i, row in chosen.iterrows():
        zrow = (row[feature_cols] - pop_mean) / pop_std
        top = zrow.abs().sort_values(ascending=False).head(3).index.tolist()
        notes = ", ".join(
            f"{c}={row[c]:.2f} ({'+' if zrow[c]>0 else ''}{zrow[c]:.1f}σ)" for c in top
        )
        print(f"  [{i+1}] {row['patient_id']:>8s}  "
              f"n={int(row['n_readings']):>5d}  "
              f"days={row['days_span']:>5.1f}  "
              f"mean={row['mean']:>5.1f}  "
              f"std={row['std']:>5.1f}  "
              f"kurt={row['kurt']:+5.2f}  "
              f"cad_cv={row['cadence_cv']:>4.1f}  "
              f"gap>6h={row['frac_gaps_gt_6h']*100:>4.1f}%")
        print(f"      most distinctive: {notes}")
    return chosen["patient_id"].tolist()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    ids = main(args.k)
    print("\n# To train the VAE on this set:")
    print("python3 -m pipeline.smoothing.train_vae " + " ".join(ids))
