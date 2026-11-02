"""Per-patient raw and smoothed values of the quantities the signal metrics are ratios of.

metrics_per_patient.csv reports each metric as one composite number. Several of those
metrics are a ratio or difference of a raw quantity and its smoothed counterpart, which
can be plotted side by side the way mean HRV already is:

    variance_reduction  = 100 (1 - Var(x_hat) / Var(x))      -> Var(x)        vs Var(x_hat)
    roughness_reduction = 100 (1 - mean|dx_hat| / mean|dx|)  -> mean|dx|      vs mean|dx_hat|
    mean_error_pct      = 100 |mean(x_hat) - mean(x)|/|mean(x)| -> mean(x)    vs mean(x_hat)
    peak_preservation   = median_p clip(exc_hat_p / exc_p)    -> median exc   vs median exc_hat

correlation and residual_std have no such pair: both are intrinsically joint.

Masking mirrors wearable/metrics/*.py exactly — finite in BOTH columns, patients taken
in file order, population variance (ddof=0) — so the ratios reproduce the published
per-patient metrics. scripts/paired_components_check.py asserts that.

Usage: python scripts/paired_components.py <runs.csv> <out.csv>
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from os import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

WINDOW = 6          # must match wearable/metrics/peak_preservation.py


def _peak_excursions(raw, smoothed, win=WINDOW):
    """Median raw peak excursion and the smoothed excursion over the same peaks."""
    peaks, _ = find_peaks(raw, prominence=0.5 * np.nanstd(raw))
    exc, exc_hat = [], []
    for p in peaks:
        a, b = max(0, p - win), min(len(raw), p + win + 1)
        e = raw[p] - np.min(raw[a:b])
        if e > 1e-9:
            exc.append(e)
            exc_hat.append(smoothed[p] - np.min(smoothed[a:b]))
    if not exc:
        return np.nan, np.nan
    return float(np.median(exc)), float(np.median(exc_hat))


def _patient(raw, smoothed):
    raw, smoothed = np.asarray(raw, float), np.asarray(smoothed, float)
    m = np.isfinite(raw) & np.isfinite(smoothed)
    if m.sum() < 3:
        return None
    raw, smoothed = raw[m], smoothed[m]
    exc, exc_hat = _peak_excursions(raw, smoothed)
    return {
        "n": int(m.sum()),
        "mean_raw": float(np.mean(raw)), "mean_smoothed": float(np.mean(smoothed)),
        "var_raw": float(np.var(raw)), "var_smoothed": float(np.var(smoothed)),
        "std_raw": float(np.std(raw)), "std_smoothed": float(np.std(smoothed)),
        "rough_raw": float(np.mean(np.abs(np.diff(raw)))),
        "rough_smoothed": float(np.mean(np.abs(np.diff(smoothed)))),
        "peak_exc_raw": exc, "peak_exc_smoothed": exc_hat,
    }


def _run(args):
    method, variant, run_cohort, path = args
    df = pd.read_csv(path, usecols=["p_id", "hrvValue", "smoothed_hrv"])
    rows = []
    for pid, g in df.groupby("p_id", sort=True):
        r = _patient(g["hrvValue"], g["smoothed_hrv"])
        if r is not None:
            rows.append({"method": method, "variant": variant,
                         "run_cohort": run_cohort, "p_id": str(pid), **r})
    return rows


def main(runs_csv, out_csv):
    runs = pd.read_csv(runs_csv)
    # runs.csv records dir as data/runs/...; the artifacts actually live under data-2/
    jobs = [(r.method, r.variant, r.cohort,
             Path("data-2") / str(r.dir).replace("data/", "", 1) / "smoothed.csv")
            for r in runs.itertuples()]
    missing = [j[3] for j in jobs if not j[3].exists()]
    if missing:
        sys.exit(f"missing artifacts: {missing[:3]} ({len(missing)} total)")

    out = []
    with ProcessPoolExecutor(max_workers=max(1, (cpu_count() or 2) - 1)) as ex:
        for i, rows in enumerate(ex.map(_run, jobs), 1):
            out.extend(rows)
            print(f"[{i}/{len(jobs)}] {rows[0]['method']} {rows[0]['variant']} "
                  f"{rows[0]['run_cohort']}: {len(rows)} patients", flush=True)
    pd.DataFrame(out).to_csv(out_csv, index=False)
    print(f"-> {out_csv} ({len(out)} rows)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
