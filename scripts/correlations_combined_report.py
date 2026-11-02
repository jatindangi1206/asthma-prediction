"""Pearson and Spearman on one graph, per cohort group.

Reads the per-patient correlations from scripts/correlations.py, whose pearson column
is asserted to reproduce wearable/metrics/correlation.py.

Two ways of measuring the same thing: linear association, and monotone association on
ranks. Each carries a distinct marker shape as well as a colour, so identity never
rests on colour alone.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("data/reports/correlations/combined")
PER_PATIENT = Path("data/reports/correlations/correlations_per_patient.csv")
INFO = Path("data-2/reports/smoothing/metrics_per_patient.csv")

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
# measure -> (label, colour, marker). Shape is the secondary encoding.
MEASURES = {"pearson": ("Pearson r", "#2a78d6", "o"),
            "spearman": ("Spearman \u03c1", "#eb6834", "s")}

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


def select(d, group):
    """Rows for one group: its patients, under its own derivation."""
    sub = d[(d.variant == "derived") & (d.run_cohort == GROUP_RUN[group])]
    return sub[GROUPS[group](sub)]


def table(df):
    agg = {m: (m, "mean") for m in MEASURES}
    return df.groupby("method").agg(patients=("pearson", "count"), **agg).sort_index()


def plot(t, group, param_label, path, xlim):
    order = t.sort_values("pearson", ascending=False)
    y = list(range(len(order)))
    cols = list(MEASURES)

    fig, ax = plt.subplots(figsize=(7.8, 5.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # connector spans the four measures for that method, so the spread is visible
    ax.hlines(y, order[cols].min(axis=1), order[cols].max(axis=1),
              color=AXIS, lw=2, zorder=2)
    for i, m in enumerate(cols):
        label, colour, marker = MEASURES[m]
        ax.plot(order[m], y, marker, ms=8, color=colour, mec=SURFACE, mew=1.5,
                ls="none", zorder=10 - i, label=label)   # pearson stays on top

    ax.set_yticks(y, order.index, fontsize=9, color=INK)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.7, len(order) + 0.5)
    ax.set_xlabel("correlation with the raw signal — mean over patients",
                  fontsize=9, color=INK2)
    ax.set_title(f"{GROUP_LABEL[group]} — Pearson vs Spearman", fontsize=11.5,
                 color=INK, loc="left", pad=30)
    ax.annotate(f"{param_label} · {int(t.patients.max())} patients",
                (0, 1), xycoords="axes fraction", xytext=(0, 7),
                textcoords="offset points", fontsize=9, color=MUTED)
    # top rows are the weakest methods, so the upper right is free of marks
    leg = ax.legend(loc="upper right", frameon=False, fontsize=8.5,
                    handletextpad=0.3, borderpad=0.2, labelspacing=0.4)
    for txt in leg.get_texts():
        txt.set_color(INK2)
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

    md = ["# Correlation: Pearson and Spearman\n",
          "Both correlations between raw and smoothed HRV, on one graph, mean across "
          "patients. Sorted by Pearson; the connector is the gap between the two.\n",
          "Each group is smoothed with its own cohort's derived parameters; AIIMS "
          "adults and children inherit the AIIMS set and exacerbation patients take the "
          "combined set, because the study derived only three (`aiims`, `copd`, "
          "`combined`).\n",
          "Both are maximised by the identity — a smoother returning its input "
          "unchanged scores 1.0 on each — so there is no reference line.\n"]

    for group in GROUPS:
        t = table(select(d, group))
        vals = pd.concat([t[m] for m in MEASURES])
        pad = (vals.max() - vals.min()) * 0.08 or 0.05
        # extra right margin so the legend never overlaps the rightmost marker
        xlim = (vals.min() - pad, vals.max() + pad * 2.6)

        md += [f"\n## {GROUP_LABEL[group]} — {RUN_LABEL[GROUP_RUN[group]]}\n",
               "| Method | " + " | ".join(MEASURES[m][0] for m in MEASURES) + " |",
               "| --- | --- | --- |"]
        md += [f"| {m} | " + " | ".join(f"{t.loc[m, c]:.3f}" for c in MEASURES) + " |"
               for m in t.index]
        md.append(f"\n{int(t.patients.max())} patients.")

        OUT.mkdir(parents=True, exist_ok=True)
        t.to_csv(OUT / f"{group}.csv", float_format="%.5f")
        plot(t, group, RUN_LABEL[GROUP_RUN[group]], OUT / f"{group}.png", xlim)

    (OUT / "README.md").write_text("\n".join(md) + "\n")
    print(f"wrote {len(list(OUT.rglob('*.png')))} graphs to {OUT}")


if __name__ == "__main__":
    main()
