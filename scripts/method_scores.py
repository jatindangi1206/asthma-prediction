"""Score the smoothers from the metric definitions, without inventing a weighting.

A smoother must do two opposing things at once: remove noise, and keep signal. Each has
a degenerate maximiser — the identity keeps everything and removes nothing, a constant
removes everything and keeps nothing — so neither group can be read alone.

    R (remove) = sqrt( variance_reduction/100 x roughness_reduction/100 )
    K (keep)   = sqrt( peak_preservation x correlation )
    S (joint)  = sqrt( R x K )

The geometric mean is not a free choice here: it is zero when either factor is zero, so
it rejects both degenerate solutions, which an arithmetic mean does not (a constant
output would score 0.5).

Level bias (mean_error_pct) is kept OUT of S. It is not a trade-off: the noise a
smoother removes is zero-mean, so any shift in the level is pure error with a true
optimum of 0. It is reported alongside S as a gate.

Usage: python scripts/method_scores.py [--markdown]
"""
import sys

import numpy as np
import pandas as pd

GROUPS = ["aiims_adult", "aiims_paed", "copd", "aiims", "combined",
          "exacerbation_patients"]
LABEL = {"aiims_adult": "AIIMS adults", "aiims_paed": "AIIMS children", "copd": "COPD",
         "aiims": "AIIMS (all)", "combined": "Combined",
         "exacerbation_patients": "Exacerbation patients"}


def scores(group):
    t = pd.read_csv(f"data/reports/radar/{group}.csv", index_col=0)
    bias = pd.read_csv(f"data/reports/metrics_by_cohort/mean_error_pct/{group}.csv",
                       index_col=0)["mean"]
    r = np.sqrt(t.s_variance * t.s_roughness)
    k = np.sqrt(t.s_peak * t.s_corr)
    return pd.DataFrame({"R_remove": r, "K_keep": k, "S_joint": np.sqrt(r * k),
                         "bias_pct": bias})


def frontier(d):
    """Pareto frontier on R up, K up, bias down."""
    out = []
    for m in d.index:
        a = d.loc[m]
        dominated = any(
            (d.loc[o, "R_remove"] >= a.R_remove and d.loc[o, "K_keep"] >= a.K_keep
             and d.loc[o, "bias_pct"] <= a.bias_pct)
            and (d.loc[o, "R_remove"] > a.R_remove or d.loc[o, "K_keep"] > a.K_keep
                 or d.loc[o, "bias_pct"] < a.bias_pct)
            for o in d.index if o != m)
        out.append(not dominated)
    return pd.Series(out, index=d.index)


def main(markdown=False):
    per_group = {g: scores(g) for g in GROUPS}
    s = pd.DataFrame({g: d.S_joint for g, d in per_group.items()})
    b = pd.DataFrame({g: d.bias_pct for g, d in per_group.items()})

    d = per_group["combined"].copy()
    d["pareto"] = frontier(d)
    d = d.sort_values("S_joint", ascending=False)

    if markdown:
        print("| Method | R (remove) | K (keep) | S (joint) | Level bias % | Pareto |")
        print("| --- | --- | --- | --- | --- | --- |")
        for m, r in d.iterrows():
            print(f"| {m} | {r.R_remove:.3f} | {r.K_keep:.3f} | {r.S_joint:.3f} | "
                  f"{r.bias_pct:.2f} | {'yes' if r.pareto else 'no'} |")
    else:
        print("=== combined cohort ===")
        print(d.round(4).to_string())
        print("\n=== S_joint across cohorts ===")
        print(s.round(3).to_string())
        print("\n=== level bias % across cohorts ===")
        print(b.round(2).to_string())
        print("\nbest S_joint per cohort:", dict(s.idxmax()))
        print("lowest bias per cohort: ", dict(b.idxmin()))
        print(f"\nrspf bias range: {b.loc['rspf'].min():.2f}–{b.loc['rspf'].max():.2f}%; "
              f"next best any cohort: {b.drop('rspf').min().min():.2f}%")


if __name__ == "__main__":
    main("--markdown" in sys.argv)
