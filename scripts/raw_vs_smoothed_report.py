"""Raw vs smoothed, per smoother, per cohort group, for the derived settings.

Plots each quantity the signal metrics are built from as a raw/smoothed pair, so the
composite metric in data/reports/metrics_by_cohort/ can be read against the two numbers
it is a ratio of. Driven by scripts/paired_components.py output, which
scripts/paired_components_check.py asserts reproduces the published metrics.

Settings (same names as scripts/smoothing_study.py):
  derived_own       each cohort smoothed with parameters derived from itself
  derived_combined  parameters derived from the combined cohort
Default parameters are deliberately not plotted.

Patient groups follow summarise_metrics() in scripts/smoothing_study.py.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("data/reports/raw_vs_smoothed")
COMPONENTS = OUT / "components_per_patient.csv"
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

# quantity -> (label, raw column, smoothed column, the metric it feeds)
QUANTITIES = {
    "mean": ("Mean HRV", "mean_raw", "mean_smoothed", "mean_error_pct"),
    "variance": ("Variance", "var_raw", "var_smoothed", "variance_reduction"),
    "std": ("Standard deviation", "std_raw", "std_smoothed", None),
    "roughness": ("Mean step-to-step change |Δ|", "rough_raw", "rough_smoothed",
                  "roughness_reduction"),
    "peak_excursion": ("Median peak excursion", "peak_exc_raw", "peak_exc_smoothed",
                       "peak_preservation"),
}


def select(d, group):
    """Rows for one group: its patients, under its own derivation."""
    sub = d[(d.variant == "derived") & (d.run_cohort == GROUP_RUN[group])]
    return sub[GROUPS[group](sub)]


def table(df, raw_col, smoothed_col):
    """Mean across patients of each patient's raw / smoothed value."""
    t = df.groupby("method").agg(raw=(raw_col, "mean"), smoothed=(smoothed_col, "mean"),
                                 patients=(raw_col, "count"))
    t["difference"] = t.smoothed - t.raw
    return t.sort_index()


def plot(t, quantity, group, param_label, path, xlim):
    label, _, _, feeds = QUANTITIES[quantity]
    # ponytail: raw differs between methods only where a method left rows unsmoothed,
    # so the median across methods is the raw value for the patient set
    ref = t.raw.median()
    order = t.sort_values("smoothed")
    y = range(len(order))

    fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.axvline(ref, color=INK2, lw=2, ls=(0, (5, 3)), zorder=2)
    # delta stem: raw -> smoothed, so the gap is what the smoother changed
    ax.hlines(list(y), ref, order.smoothed, color=AXIS, lw=2, zorder=3)
    ax.plot(order.smoothed, list(y), "o", ms=9, color=SERIES,
            mec=SURFACE, mew=2, ls="none", zorder=4)

    fmt = "{:.3f}" if order.smoothed.abs().max() < 2 else "{:.2f}"
    for i, v in zip(y, order.smoothed):
        ax.annotate(fmt.format(v), (v, i), xytext=(8 if v >= ref else -8, 0),
                    textcoords="offset points", va="center",
                    ha="left" if v >= ref else "right", fontsize=8.5, color=INK)

    ax.annotate(f"raw {fmt.format(ref)}", (ref, len(order) - 0.3), xytext=(6, 0),
                textcoords="offset points", fontsize=8.5, color=INK2, fontweight="bold")

    ax.set_yticks(list(y), order.index, fontsize=9, color=INK)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.7, len(order) - 0.1)
    ax.set_xlabel(f"{label} — averaged over patients", fontsize=9, color=INK2)
    ax.set_title(f"{GROUP_LABEL[group]} — raw vs smoothed {label.lower()}",
                 fontsize=11.5, color=INK, loc="left", pad=42)
    if feeds:
        ax.annotate(f"the two terms of  {feeds}", (0, 1), xycoords="axes fraction",
                    xytext=(0, 24), textcoords="offset points", fontsize=9.5, color=INK2)
    ax.annotate(f"{param_label} · {int(t.patients.max())} patients",
                (0, 1), xycoords="axes fraction", xytext=(0, 7),
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
    d = pd.read_csv(COMPONENTS)
    info = (pd.read_csv(INFO, usecols=["p_id", "cohort", "group", "has_exac"])
              .drop_duplicates("p_id").set_index("p_id"))
    d = d.merge(info, left_on="p_id", right_index=True, how="left")
    assert d.cohort.notna().all(), "patients missing a cohort label"

    md = ["# Raw vs smoothed, by cohort group\n",
          "Each quantity the signal metrics are a ratio of, plotted as the raw value "
          "(dashed line) against each smoother's value. Mean across patients, paired on "
          "the rows where the smoother produced an estimate.\n",
          "**Each group is smoothed with its own cohort's derived parameters** — COPD "
          "under COPD-derived settings, the combined group under settings derived from "
          "all 187 patients. The study derived three parameter sets (`aiims`, `copd`, "
          "`combined`), so AIIMS adults and AIIMS children both inherit the AIIMS set, "
          "and exacerbation patients, who span both cohorts, take the combined set. "
          "Default parameters are not plotted.\n",
          "`correlation` and `residual_std` have no raw/smoothed pair — both are "
          "intrinsically joint, so they appear only in "
          "`data/reports/metrics_by_cohort/`.\n"]

    for quantity, (label, raw_col, smoothed_col, feeds) in QUANTITIES.items():
        md.append(f"\n## {label}\n"
                  + (f"\nThe two terms of `{feeds}`.\n" if feeds else ""))
        for group in GROUPS:
            t = table(select(d, group), raw_col, smoothed_col)
            vals = pd.concat([t.smoothed, t.raw])
            pad = (vals.max() - vals.min()) * 0.22 or abs(vals.max()) * 0.1 or 1.0
            xlim = (vals.min() - pad, vals.max() + pad * 1.4)

            md += [f"\n### {GROUP_LABEL[group]} — {RUN_LABEL[GROUP_RUN[group]]}\n",
                   "| Method | Raw | Smoothened | Difference |",
                   "| --- | --- | --- | --- |"]
            md += [f"| {m} | {r.raw:.2f} | {r.smoothed:.2f} | {r.difference:+.2f} |"
                   for m, r in t.iterrows()]
            md.append(f"\n{int(t.patients.max())} patients. "
                      f"Raw reference line: {t.raw.median():.2f}.")

            dst = OUT / quantity
            dst.mkdir(parents=True, exist_ok=True)
            t.to_csv(dst / f"{group}.csv", float_format="%.4f")
            plot(t, quantity, group, RUN_LABEL[GROUP_RUN[group]],
                 dst / f"{group}.png", xlim)

    (OUT / "README.md").write_text("\n".join(md) + "\n")
    print(f"wrote {len(list(OUT.rglob('*.png')))} graphs, "
          f"{len(list(OUT.rglob('*/*.csv')))} tables to {OUT}")


if __name__ == "__main__":
    main()
