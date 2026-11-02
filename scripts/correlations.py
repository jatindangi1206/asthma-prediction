"""Per-patient raw-vs-smoothed association under four different definitions.

wearable/metrics/correlation.py is Pearson only. Pearson answers "do they move together
linearly"; the alternatives answer different questions, and on this signal they can
disagree, because HRV is device-capped at 129 (heavy ties at the cap, which tau-b and
Spearman handle explicitly) and the smoothers shift the level (which only the
concordance coefficient penalises).

  pearson   linear association; invariant to any affine change of either series
  spearman  monotone association; invariant to any monotone change (ranks only)
  kendall   tau-b, concordant minus discordant pairs, tie-corrected
  ccc       Lin's concordance: agreement with the 45-degree line, NOT just association,
            so a constant offset or a scale change costs you

Masking mirrors wearable/metrics/*.py exactly, so the pearson column must reproduce the
published per-patient correlation; main() asserts it.

Usage: python scripts/correlations.py <runs.csv> <out.csv>
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from os import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

PUBLISHED = Path("data-2/reports/smoothing/metrics_per_patient.csv")


def _ccc(raw, smoothed):
    """Lin's concordance correlation coefficient."""
    vr, vs = np.var(raw), np.var(smoothed)
    cov = np.mean((raw - raw.mean()) * (smoothed - smoothed.mean()))
    denom = vr + vs + (raw.mean() - smoothed.mean()) ** 2
    return float("nan") if denom == 0 else float(2 * cov / denom)


def _patient(raw, smoothed):
    raw, smoothed = np.asarray(raw, float), np.asarray(smoothed, float)
    m = np.isfinite(raw) & np.isfinite(smoothed)
    if m.sum() < 3 or np.std(raw[m]) == 0 or np.std(smoothed[m]) == 0:
        return None
    raw, smoothed = raw[m], smoothed[m]
    return {
        "n": int(len(raw)),
        "pearson": float(np.corrcoef(raw, smoothed)[0, 1]),
        "spearman": float(spearmanr(raw, smoothed).statistic),
        "kendall": float(kendalltau(raw, smoothed).statistic),
        "ccc": _ccc(raw, smoothed),
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
    df = pd.DataFrame(out)

    pub = pd.read_csv(PUBLISHED, usecols=["method", "variant", "run_cohort", "p_id",
                                          "correlation"])
    keys = ["method", "variant", "run_cohort", "p_id"]
    chk = df.merge(pub, on=keys)
    err = (chk.pearson - chk.correlation).abs().max()
    print(f"pearson vs published correlation: {len(chk)} matched, max abs error {err:.3e}")
    assert err < 1e-9, "pearson does not reproduce the published correlation"

    df.to_csv(out_csv, index=False)
    print(f"-> {out_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
