"""One radar per smoother, per cohort group — a 2x5 grid of profiles for each cohort.

Spokes, their 0-1 scaling and their labels are taken unchanged from the radar already in
scripts/smoothing_dashboard.html (RADAR, ~line 592), so these PNGs and the dashboard say
the same thing. Each spoke is 0 at the centre, 1 at the edge, rings at .25/.5/.75:

    Variance        variance_reduction / 100        outward = more variance removed
    Roughness       roughness_reduction / 100       outward = more jitter removed
    Mean err        min(mean_error_pct / 10, 1)     outward = MORE level drift (worse)
    Peak score      peak_preservation               outward = peaks kept
    Corr            correlation                     outward = tracks the raw signal
    Residual share  residual_std / std_raw          outward = bigger share of the
                                                    raw spread removed

Only the Mean err spoke runs the other way; the rest are "outward = more of that
property", which is not the same as better — a shape that fills Variance and Roughness
while collapsing on Peak score is over-smoothing. The dashboard's dashed "ideal filter"
outline is deliberately NOT drawn here: it rests on the cohort noise variance R, which
is a variogram extrapolation under a level-plus-white-noise assumption (see
wearable/cohort_stats.py), and the reference lines were dropped from the other reports
for that reason.

Each group uses the most specific parameters the study derived, as elsewhere.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("data/reports/radar")
METRICS = Path("data-2/reports/smoothing/metrics_per_patient.csv")
RESIDUAL = Path("data/reports/metrics_by_cohort/residual_std_per_patient.csv")
COMPONENTS = Path("data/reports/raw_vs_smoothed/components_per_patient.csv")

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SERIES = "#e1e0d9", "#c3c2b7", "#2a78d6"

GROUPS = {
    "aiims_adult": lambda d: d.group == "aiims_adult",
    "aiims_paed": lambda d: d.group == "aiims_paed",
    "copd": lambda d: d.cohort == "copd",
    "aiims": lambda d: d.cohort == "aiims",
    "combined": lambda d: d.cohort.notna(),
    "exacerbation_patients": lambda d: d.has_exac,
}
GROUP_LABEL = {"aiims_adult": "AIIMS adults", "aiims_paed": "AIIMS children",
               "copd": "COPD", "aiims": "AIIMS (all)", "combined": "Combined",
               "exacerbation_patients": "Exacerbation patients"}
GROUP_RUN = {"aiims_adult": "aiims", "aiims_paed": "aiims", "aiims": "aiims",
             "copd": "copd", "combined": "combined",
             "exacerbation_patients": "combined"}
RUN_LABEL = {"aiims": "AIIMS-derived parameters", "copd": "COPD-derived parameters",
             "combined": "combined-derived parameters"}

# spoke label -> column on the 0-1 scale (built in load())
SPOKES = {"Variance": "s_variance", "Roughness": "s_roughness", "Mean err": "s_mean_err",
          "Peak score": "s_peak", "Corr": "s_corr", "Residual share": "s_residual"}


def load():
    keys = ["method", "variant", "run_cohort", "p_id"]
    d = pd.read_csv(METRICS)
    d = d.merge(pd.read_csv(RESIDUAL), on=keys, how="left")
    d = d.merge(pd.read_csv(COMPONENTS)[keys + ["std_raw"]], on=keys, how="left")
    # the dashboard's scalings, unchanged
    d["s_variance"] = d.variance_reduction / 100
    d["s_roughness"] = d.roughness_reduction / 100
    d["s_mean_err"] = (d.mean_error_pct / 10).clip(upper=1)
    d["s_peak"] = d.peak_preservation
    d["s_corr"] = d.correlation
    d["s_residual"] = d.residual_std / d.std_raw
    return d


def select(d, group):
    sub = d[(d.variant == "derived") & (d.run_cohort == GROUP_RUN[group])]
    return sub[GROUPS[group](sub)]


def table(df):
    t = df.groupby("method").agg(**{c: (c, "mean") for c in SPOKES.values()},
                                 patients=("s_corr", "count"))
    return t.sort_index()


def radar(t, group, path):
    labels = list(SPOKES)
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    closed = np.concatenate([ang, ang[:1]])

    fig, axes = plt.subplots(2, 5, figsize=(15, 7.2), dpi=180,
                             subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor(SURFACE)

    for ax, (method, row) in zip(axes.ravel(), t.iterrows()):
        vals = np.clip([row[SPOKES[l]] for l in labels], 0, 1)
        v = np.concatenate([vals, vals[:1]])
        ax.set_facecolor(SURFACE)
        ax.plot(closed, v, color=SERIES, lw=1.8, zorder=3)
        ax.fill(closed, v, color=SERIES, alpha=0.22, zorder=2)
        ax.plot(ang, vals, "o", ms=4, color=SERIES, zorder=4)

        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([])
        ax.set_xticks(ang)
        ax.set_xticklabels(labels, fontsize=8, color=INK2)
        ax.tick_params(pad=2)
        ax.grid(color=GRID, lw=0.8)
        ax.spines["polar"].set_color(AXIS)
        ax.set_title(method, fontsize=11, color=INK, pad=12)

    fig.suptitle(f"{GROUP_LABEL[group]} — each smoother's profile", fontsize=13,
                 color=INK, x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.945,
             f"{RUN_LABEL[GROUP_RUN[group]]} · {int(t.patients.max())} patients · "
             f"each spoke 0 at centre to 1 at edge · outward = more of that property, "
             f"not better (Mean err outward = worse)",
             fontsize=9, color=MUTED, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    d = load()
    OUT.mkdir(parents=True, exist_ok=True)

    md = ["# Smoother profiles (radar), by cohort group\n",
          "One radar per smoother, ten per cohort. Spokes and their 0-1 scalings are "
          "taken unchanged from the radar in `scripts/smoothing_dashboard.html`, so "
          "these agree with the dashboard.\n",
          "| Spoke | Value plotted | Outward means |",
          "| --- | --- | --- |",
          "| Variance | `variance_reduction / 100` | more variance removed |",
          "| Roughness | `roughness_reduction / 100` | more jitter removed |",
          "| Mean err | `min(mean_error_pct / 10, 1)` | **more** level drift — worse |",
          "| Peak score | `peak_preservation` | peaks kept |",
          "| Corr | `correlation` | tracks the raw signal |",
          "| Residual share | `residual_std / std_raw` | bigger share of the raw spread removed |",
          "\nOnly Mean err runs the other way. Outward is *more of that property*, not "
          "better: a shape filling Variance and Roughness while collapsing on Peak score "
          "is over-smoothing. The dashboard's dashed \"ideal filter\" outline is not "
          "drawn here — it rests on the cohort noise variance `R`, a variogram "
          "extrapolation under a level-plus-white-noise assumption, and the reference "
          "lines were dropped from the other reports for that reason.\n",
          "Values are the mean across patients, matching the other reports; the "
          "dashboard uses the median patient, so small differences are expected.\n"]

    for group in GROUPS:
        t = table(select(d, group))
        t.to_csv(OUT / f"{group}.csv", float_format="%.4f")
        radar(t, group, OUT / f"{group}.png")
        md += [f"\n## {GROUP_LABEL[group]} — {RUN_LABEL[GROUP_RUN[group]]}\n",
               "| Method | " + " | ".join(SPOKES) + " |",
               "| --- |" + " --- |" * len(SPOKES)]
        md += ["| " + m + " | " + " | ".join(f"{t.loc[m, SPOKES[s]]:.3f}"
                                             for s in SPOKES) + " |" for m in t.index]
        md.append(f"\n{int(t.patients.max())} patients.")

    (OUT / "README.md").write_text("\n".join(md) + "\n")
    print(f"wrote {len(list(OUT.glob('*.png')))} radars to {OUT}")


if __name__ == "__main__":
    main()
