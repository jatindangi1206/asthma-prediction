"""Assert the paired components reproduce the published per-patient metrics.

If Var(x) and Var(x_hat) are right, then 100 (1 - Var(x_hat)/Var(x)) must equal the
variance_reduction already in metrics_per_patient.csv, patient by patient. Same for
roughness_reduction and mean_error_pct. peak_preservation is NOT checked this way:
the metric takes the median of per-peak ratios, which is not the ratio of medians.
"""
import sys

import numpy as np
import pandas as pd

KEYS = ["method", "variant", "run_cohort", "p_id"]


def main(components_csv):
    c = pd.read_csv(components_csv)
    m = pd.read_csv("data-2/reports/smoothing/metrics_per_patient.csv")
    d = c.merge(m, on=KEYS, suffixes=("", "_pub"))
    print(f"{len(d)} patient-runs matched")

    checks = {
        "variance_reduction": 100 * (1 - d.var_smoothed / d.var_raw),
        "roughness_reduction": 100 * (1 - d.rough_smoothed / d.rough_raw),
        "mean_error_pct": 100 * (d.mean_smoothed - d.mean_raw).abs() / d.mean_raw.abs(),
    }
    bad = 0
    for name, rebuilt in checks.items():
        pub = d[name]
        ok = pub.notna() & rebuilt.notna()
        err = (rebuilt[ok] - pub[ok]).abs().max()
        print(f"{name:22s} n={int(ok.sum()):5d}  max abs error {err:.3e}")
        bad += err > 1e-6
    print("MISMATCH" if bad else "all components reproduce the published metrics")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
