"""Every signal metric, per smoother, per cohort group, for the derived settings.

The companion to scripts/raw_vs_smoothed_report.py: same cohort groups, same two
settings, same dot-plot form.

Reference lines are deliberately sparse. Only mean_error_pct gets one, at 0: noise is
zero-mean, so a smoother has no business moving the level. Every other metric is
plotted as a ranking with no reference, because the only available benchmarks (the
cohort noise variance R, and the ideal_reference Kalman yardstick built on it) are
estimated by extrapolating a variogram to zero lag under a level-plus-white-noise
assumption, and R absorbs any sub-10-min physiology. See wearable/cohort_stats.py.

Per-patient inputs: data-2/reports/smoothing/metrics_per_patient.csv for five metrics,
plus residual_std computed separately (the report's per-patient file predates it).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("data/reports/metrics_by_cohort")
PER_PATIENT = Path("data-2/reports/smoothing/metrics_per_patient.csv")

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
# Reference line per metric, or None for a plain ranking. Chosen explicitly:
# a smoother removes zero-mean noise, so it must not shift the level.
REFERENCE = {"mean_error_pct": (0.0, "ideal 0 — no level shift")}

# formula as matplotlib mathtext, transcribed from wearable/metrics/<name>.py
METRICS = {
    "correlation": (
        "Correlation (raw vs smoothed)", True,
        r"$r=\mathrm{corr}(x,\hat{x})$"),
    "mean_error_pct": (
        "Mean error %", False,
        r"$100\cdot|\bar{\hat{x}}-\bar{x}|\,/\,|\bar{x}|$"),
    "peak_preservation": (
        "Peak preservation (0-1)", True,
        r"$\mathrm{median}_p\ \mathrm{clip}\left(\frac{\hat{x}_p-\min\,\hat{x}_{[p\pm 6]}}"
        r"{x_p-\min\,x_{[p\pm 6]}},0,1\right)$"),
    "residual_std": (
        "Residual std", False,
        r"$\mathrm{std}(x-\hat{x})$"),
    "roughness_reduction": (
        "Roughness reduction %", True,
        r"$100\left(1-\overline{|\Delta\hat{x}|}\,/\,\overline{|\Delta x|}\right)$"),
    "variance_reduction": (
        "Variance reduction %", True,
        r"$100\left(1-\mathrm{Var}(\hat{x})\,/\,\mathrm{Var}(x)\right)$"),
}


def select(d, group):
    """Rows for one group: its patients, under its own derivation."""
    sub = d[(d.variant == "derived") & (d.run_cohort == GROUP_RUN[group])]
    return sub[GROUPS[group](sub)]


def table(df, metric):
    t = df.groupby("method").agg(mean=(metric, "mean"), median=(metric, "median"),
                                 patients=(metric, "count"))
    return t.sort_index()


def plot(t, metric, group, param_label, path, xlim):
    name, higher_better, formula = METRICS[metric]
    ref = REFERENCE.get(metric)
    order = t.sort_values("mean", ascending=not higher_better)
    y = range(len(order))

    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # hairline row guide, not a magnitude encoding: there is no zero baseline here
    ax.hlines(list(y), xlim[0], order["mean"], color=GRID, lw=1, zorder=2)
    if ref is not None:
        ax.axvline(ref[0], color=INK2, lw=2, ls=(0, (5, 3)), zorder=3)
    ax.plot(order["mean"], list(y), "o", ms=9, color=SERIES,
            mec=SURFACE, mew=2, ls="none", zorder=4)

    fmt = "{:.3f}" if order["mean"].abs().max() < 2 else "{:.2f}"
    for i, v in zip(y, order["mean"]):
        ax.annotate(fmt.format(v), (v, i), xytext=(8, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8.5, color=INK)

    if ref is not None:
        ax.annotate(ref[1], (ref[0], len(order) - 0.3), xytext=(6, 0),
                    textcoords="offset points", fontsize=8.5, color=INK2,
                    fontweight="bold")

    ax.set_yticks(list(y), order.index, fontsize=9, color=INK)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.7, len(order) - 0.1)
    ax.set_xlabel(f"{name} — mean over patients"
                  f" ({'higher' if higher_better else 'lower'} is better)",
                  fontsize=9, color=INK2)
    ax.set_title(f"{GROUP_LABEL[group]} — {name}", fontsize=11.5,
                 color=INK, loc="left", pad=42)
    ax.annotate(formula, (0, 1), xycoords="axes fraction", xytext=(0, 24),
                textcoords="offset points", fontsize=11, color=INK)
    ax.annotate(f"{param_label} · {int(t.patients.max())} patients",
                (0, 1), xycoords="axes fraction", xytext=(0, 7),
                textcoords="offset points", fontsize=9, color=MUTED)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main(residual_csv):
    d = pd.read_csv(PER_PATIENT)
    rs = pd.read_csv(residual_csv)
    d = d.merge(rs, on=["method", "variant", "run_cohort", "p_id"], how="left")
    assert d.residual_std.notna().any(), "residual_std join produced nothing"

    md = ["# Signal metrics by cohort group\n",
          "Mean across patients, for each smoother and cohort group.\n",
          "**Each group is smoothed with its own cohort's derived parameters** — COPD "
          "under COPD-derived settings, the combined group under settings derived from "
          "all 187 patients. The study derived three parameter sets (`aiims`, `copd`, "
          "`combined`), so AIIMS adults and AIIMS children both inherit the AIIMS set, "
          "and exacerbation patients, who span both cohorts, take the combined set. "
          "Default parameters are not plotted.\n",
          "**Reference lines.** Only `mean_error_pct` carries one, at 0: a smoother "
          "removes zero-mean noise, so it should not shift the level at all. The other "
          "five metrics are plotted as rankings with no reference, because every "
          "available benchmark — the cohort noise variance `R`, and the "
          "`ideal_reference` Kalman yardstick built on it — is estimated by "
          "extrapolating a robust variogram to zero lag under a "
          "level-plus-white-noise assumption, and `R` absorbs any sub-10-min "
          "physiology along with sensor error. Compare methods against each other "
          "rather than against a modelled optimum.\n",
          "Each CSV also carries the median, which is what "
          "`data/reports/smoothing/metrics_summary.csv` reports.\n"]

    for metric, (name, higher_better, formula) in METRICS.items():
        md.append(f"\n## {name}\n\n`{metric}` — "
                  f"{'higher' if higher_better else 'lower'} is better"
                  + (" · reference line at 0" if metric in REFERENCE else
                     " · no reference line") + "\n")
        for group in GROUPS:
            t = table(select(d, group), metric)
            lo, hi = t["mean"].min(), t["mean"].max()
            if metric in REFERENCE:          # keep the line itself in frame
                lo, hi = min(lo, REFERENCE[metric][0]), max(hi, REFERENCE[metric][0])
            pad = (hi - lo) * 0.22 or abs(hi) * 0.1 or 1.0
            # right pad leaves room for value labels; a metric sitting on its reference
            # gets only a hairline margin, so no impossible negative tick appears
            left = pad * (0.22 if metric in REFERENCE else 1.0)
            xlim = (lo - left, hi + pad * 1.6)

            md += [f"\n### {GROUP_LABEL[group]} — {RUN_LABEL[GROUP_RUN[group]]}\n",
                   "| Method | Mean | Median |", "| --- | --- | --- |"]
            md += [f"| {m} | {r['mean']:.3f} | {r['median']:.3f} |"
                   for m, r in t.iterrows()]
            md.append(f"\n{int(t.patients.max())} patients.")

            dst = OUT / metric
            dst.mkdir(parents=True, exist_ok=True)
            t.to_csv(dst / f"{group}.csv", float_format="%.5f")
            plot(t, metric, group, RUN_LABEL[GROUP_RUN[group]],
                 dst / f"{group}.png", xlim)

    (OUT / "README.md").write_text("\n".join(md) + "\n")
    print(f"wrote {len(list(OUT.rglob('*.png')))} graphs, "
          f"{len(list(OUT.rglob('*/*.csv')))} tables to {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
