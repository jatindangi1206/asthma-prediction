"""Raw-vs-smoothed association under four definitions, per cohort group and setting.

Same ranking form as data/reports/metrics_by_cohort/: no reference line. All four
measures are maximised by doing nothing at all (the identity scores 1.0 on every one of
them), so there is no benchmark to draw — compare the methods against each other.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("data/reports/correlations")
PER_PATIENT = OUT / "correlations_per_patient.csv"
INFO = Path("data-2/reports/smoothing/metrics_per_patient.csv")

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

# Each group is scored under the most specific parameters the study actually derived.
# settings.json holds three: aiims, copd, combined. There is no aiims_adult, aiims_paed
# or exacerbation_patients derivation, so those groups inherit their parent cohort's,
# and exacerbation patients span both cohorts so they take the combined one.
GROUP_RUN = {"aiims_adult": "aiims", "aiims_paed": "aiims", "aiims": "aiims",
             "copd": "copd", "combined": "combined",
             "exacerbation_patients": "combined"}
RUN_LABEL = {"aiims": "AIIMS-derived parameters", "copd": "COPD-derived parameters",
             "combined": "combined-derived parameters"}

# name -> (label, formula as mathtext, what it is invariant to)
MEASURES = {
    "pearson": (
        "Pearson r",
        r"$r=\frac{\sum(x-\bar{x})(\hat{x}-\bar{\hat{x}})}"
        r"{\sqrt{\sum(x-\bar{x})^2\sum(\hat{x}-\bar{\hat{x}})^2}}$",
        "linear association · blind to any shift or rescaling of either series"),
    "spearman": (
        "Spearman ρ",
        r"$\rho=\mathrm{corr}\left(\mathrm{rank}(x),\ \mathrm{rank}(\hat{x})\right)$",
        "monotone association · ranks only, so blind to any monotone change"),
}


def select(d, group):
    """Rows for one group: its patients, under its own derivation."""
    sub = d[(d.variant == "derived") & (d.run_cohort == GROUP_RUN[group])]
    return sub[GROUPS[group](sub)]


def table(df, measure):
    return df.groupby("method").agg(mean=(measure, "mean"), median=(measure, "median"),
                                    patients=(measure, "count")).sort_index()


def plot(t, measure, group, param_label, path, xlim):
    label, formula, note = MEASURES[measure]
    order = t.sort_values("mean", ascending=False)
    y = range(len(order))

    fig, ax = plt.subplots(figsize=(7.4, 5.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # hairline row guide, not a magnitude encoding: there is no zero baseline here
    ax.hlines(list(y), xlim[0], order["mean"], color=GRID, lw=1, zorder=2)
    ax.plot(order["mean"], list(y), "o", ms=9, color=SERIES,
            mec=SURFACE, mew=2, ls="none", zorder=4)
    for i, v in zip(y, order["mean"]):
        ax.annotate(f"{v:.3f}", (v, i), xytext=(8, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8.5, color=INK)

    ax.set_yticks(list(y), order.index, fontsize=9, color=INK)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.7, len(order) - 0.1)
    ax.set_xlabel(f"{label} — mean over patients (higher = closer to the raw signal)",
                  fontsize=9, color=INK2)
    ax.set_title(f"{GROUP_LABEL[group]} — {label}", fontsize=11.5,
                 color=INK, loc="left", pad=68)
    ax.annotate(formula, (0, 1), xycoords="axes fraction", xytext=(0, 36),
                textcoords="offset points", fontsize=11, color=INK)
    ax.annotate(note, (0, 1), xycoords="axes fraction", xytext=(0, 20),
                textcoords="offset points", fontsize=8.5, color=INK2)
    ax.annotate(f"{param_label} · {int(t.patients.max())} patients",
                (0, 1), xycoords="axes fraction", xytext=(0, 6),
                textcoords="offset points", fontsize=9, color=MUTED)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    d = pd.read_csv(PER_PATIENT)
    info = (pd.read_csv(INFO, usecols=["p_id", "cohort", "group", "has_exac"])
              .drop_duplicates("p_id").set_index("p_id"))
    d = d.merge(info, left_on="p_id", right_index=True, how="left")
    assert d.cohort.notna().all(), "patients missing a cohort label"

    md = ["# Raw vs smoothed association: Pearson and Spearman\n",
          "Mean across patients, for each smoother and cohort group. Each group is "
          "smoothed with its own cohort's derived parameters; AIIMS adults and children "
          "inherit the AIIMS set and exacerbation patients take the combined set, "
          "because the study derived only three (`aiims`, `copd`, `combined`).\n",
          "**No reference line.** Every measure here is maximised by the identity: a "
          "smoother that returns its input unchanged scores 1.0 on both. They "
          "therefore say whether a smoother kept hold of the signal, not whether it "
          "cleaned it — read them next to `peak_preservation` and the raw-vs-smoothed "
          "variance graphs.\n",
          "`pearson` reproduces `wearable/metrics/correlation.py` exactly (asserted in "
          "scripts/correlations.py); `spearman` is computed there too.\n"]

    for measure, (label, formula, note) in MEASURES.items():
        md.append(f"\n## {label}\n\n{note}\n")
        for group in GROUPS:
            t = table(select(d, group), measure)
            pad = (t["mean"].max() - t["mean"].min()) * 0.22 or 0.05
            xlim = (t["mean"].min() - pad, t["mean"].max() + pad * 1.6)

            md += [f"\n### {GROUP_LABEL[group]} — {RUN_LABEL[GROUP_RUN[group]]}\n",
                   "| Method | Mean | Median |", "| --- | --- | --- |"]
            md += [f"| {m} | {r['mean']:.3f} | {r['median']:.3f} |"
                   for m, r in t.iterrows()]
            md.append(f"\n{int(t.patients.max())} patients.")

            dst = OUT / measure
            dst.mkdir(parents=True, exist_ok=True)
            t.to_csv(dst / f"{group}.csv", float_format="%.5f")
            plot(t, measure, group, RUN_LABEL[GROUP_RUN[group]],
                 dst / f"{group}.png", xlim)

    (OUT / "README.md").write_text("\n".join(md) + "\n")
    print(f"wrote {len(list(OUT.rglob('*.png')))} graphs, "
          f"{len(list(OUT.rglob('*/*.csv')))} tables to {OUT}")


if __name__ == "__main__":
    main()
