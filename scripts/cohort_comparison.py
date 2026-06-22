"""Cohort-level smoother comparison (all methods × all patients).

The per-patient report (`smoother_comparison_report.py`) emits one HTML per
patient — unusable for comparing 10 methods across ~109 patients. This script
instead reads every ALREADY-computed `data/smoothed_<method>/` CSV (no
re-running of any smoother), scores each patient with the same metrics the
per-patient report uses, and aggregates the *distributions* across the cohort.

Outputs (from asthma-prediction/):
  data/results/cohort_comparison_per_patient.csv   long table: one row per (method, patient)
  data/results/cohort_comparison_summary.csv       wide table: per-method median (IQR) of each metric
  data/plots/cohort_comparison.html                ranked summary table + per-metric boxplots

Run:
    python scripts/cohort_comparison.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import plotly.graph_objects as go
from plotly.subplots import make_subplots

GAP_THRESHOLD_MIN = 180.0

# method key -> (directory, filename suffix). Mirrors SMOOTHERS in 03_annotate.py.
METHODS = [
    ("pf",        "data/smoothed",           "_smoothed"),   # non-causal (FFBS) — offline ref
    ("rspf",      "data/smoothed_rspf",      "_rspf"),
    ("krlst",     "data/smoothed_krlst",     "_krlst"),
    ("gpssm",     "data/smoothed_gpssm",     "_gpssm"),
    ("ossa",      "data/smoothed_ossa",      "_ossa"),
    ("kim",       "data/smoothed_kim",       "_kim"),
    ("gammadglm", "data/smoothed_gammadglm", "_gammadglm"),
    ("logmhw",    "data/smoothed_logmhw",    "_logmhw"),
    ("hrkf",      "data/smoothed_hrkf",      "_hrkf"),
    ("anrewma",   "data/smoothed_anrewma",   "_anrewma"),
    ("ctimm",     "data/smoothed_ctimm",     "_ctimm"),
]

# Metrics where a HIGHER value is better (drives the colour scale / ranking direction).
HIGHER_BETTER = {"var_red": True, "rough_red": True, "peak": True, "corr": True,
                 "mean_err_pct": False, "resid_std": False}
METRIC_LABEL = {
    "var_red":      "Variance ↓ %",
    "rough_red":    "Roughness ↓ %",
    "mean_err_pct": "Mean err %",
    "peak":         "Peak score",
    "corr":         "Corr",
    "resid_std":    "Residual std (ms)",
}


def peak_preservation(raw, sm, win=6):
    """Median fraction of each raw peak's local excursion retained after smoothing."""
    pk, _ = find_peaks(raw, prominence=0.5 * np.nanstd(raw))
    if pk.size == 0:
        return np.nan
    ratios, n = [], len(raw)
    for p in pk:
        a, b = max(0, p - win), min(n, p + win + 1)
        raw_exc = raw[p] - np.min(raw[a:b])
        sm_exc = sm[p] - np.min(sm[a:b])
        if raw_exc > 1e-9:
            ratios.append(np.clip(sm_exc / raw_exc, 0, 1))
    return float(np.median(ratios)) if ratios else np.nan


def score_patient(csv_path):
    """Compute comparison metrics from one smoothed CSV (observed rows only)."""
    df = pd.read_csv(csv_path)
    raw = pd.to_numeric(df.get("hrvValue"), errors="coerce").to_numpy(dtype=float)
    sm = pd.to_numeric(df.get("smoothed_hrv"), errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(raw) & np.isfinite(sm)
    raw, sm = raw[m], sm[m]
    if raw.size < 3:
        return None
    var_red = 100.0 * (1 - np.var(sm) / np.var(raw)) if np.var(raw) > 0 else np.nan
    rough_raw = np.mean(np.abs(np.diff(raw)))
    rough_red = 100.0 * (1 - np.mean(np.abs(np.diff(sm))) / rough_raw) if rough_raw > 0 else np.nan
    mean_err_pct = 100.0 * abs(np.mean(sm) - np.mean(raw)) / abs(np.mean(raw)) if np.mean(raw) else np.nan
    corr = float(np.corrcoef(raw, sm)[0, 1]) if raw.size > 1 else np.nan
    n_chunks = int(df["chunk_id"].nunique()) if "chunk_id" in df.columns else np.nan
    return dict(var_red=var_red, rough_red=rough_red, mean_err_pct=mean_err_pct,
                peak=peak_preservation(raw, sm), corr=corr,
                resid_std=float(np.std(raw - sm)), n_used=int(raw.size), n_chunks=n_chunks)


def collect():
    rows = []
    for key, d, suf in METHODS:
        files = sorted(Path(d).glob(f"*{suf}.csv"))
        if not files:
            print(f"  {key:9s}  (no files in {d})")
            continue
        ok = 0
        for f in files:
            pid = f.stem[: -len(suf)] if suf and f.stem.endswith(suf) else f.stem
            try:
                mt = score_patient(f)
            except Exception as ex:
                mt = None
                print(f"    {key} {pid}: {type(ex).__name__}: {ex}")
            if mt:
                rows.append(dict(method=key, pid=pid, **mt))
                ok += 1
        print(f"  {key:9s}  scored {ok}/{len(files)} patients")
    return pd.DataFrame(rows)


def summarize(long):
    """Per-method median and IQR for each metric, plus mean coverage/chunks."""
    metrics = list(METRIC_LABEL)
    recs = []
    for key, _, _ in METHODS:
        sub = long[long["method"] == key]
        if sub.empty:
            continue
        rec = {"method": key, "n_patients": len(sub)}
        for mt in metrics:
            v = sub[mt].dropna()
            rec[f"{mt}_med"] = float(v.median()) if len(v) else np.nan
            rec[f"{mt}_iqr"] = float(v.quantile(0.75) - v.quantile(0.25)) if len(v) else np.nan
        rec["chunks_med"] = float(sub["n_chunks"].median())
        recs.append(rec)
    return pd.DataFrame(recs)


def build_html(long, summ, out_path):
    metrics = list(METRIC_LABEL)
    # ---- ranked summary table: rank by median var_red (primary smoothness proxy) ----
    order = summ.sort_values("var_red_med", ascending=False)["method"].tolist()
    th = ["Method", "Patients"] + [METRIC_LABEL[m] + "  median (IQR)" for m in metrics] + ["Chunks med"]
    trs = []
    for key in order:
        r = summ[summ["method"] == key].iloc[0]
        cells = [f"<b>{key}</b>", f"{int(r['n_patients'])}"]
        for mt in metrics:
            cells.append(f"{r[f'{mt}_med']:.2f} <span style='color:#888'>({r[f'{mt}_iqr']:.2f})</span>")
        cells.append(f"{r['chunks_med']:.0f}")
        trs.append("<tr>" + "".join(f"<td style='padding:5px 9px'>{c}</td>" for c in cells) + "</tr>")
    table = (
        "<h2 style='font-family:Arial'>Cohort smoother comparison — "
        f"{long['pid'].nunique()} patients × {long['method'].nunique()} methods</h2>"
        "<p style='font-family:Arial;color:#555;max-width:1100px'>Each smoother's "
        "<code>smoothed_hrv</code> is scored per patient on observed rows, then aggregated across the "
        "cohort. Cells show the <b>median</b> across patients with the <b>inter-quartile range</b> in "
        "parentheses (spread = how consistent the method is across people). Rows are ranked by median "
        "variance reduction. ↓ metrics: higher = smoother. Peak score &amp; Corr near 1 = better fidelity. "
        "<b>pf</b> is the non-causal FFBS reference (offline only).</p>"
        "<table border='1' cellspacing='0' cellpadding='0' style='border-collapse:collapse;"
        "font-family:Arial;font-size:13px'><thead style='background:#1f3864;color:#fff'><tr>"
        + "".join(f"<th style='padding:6px 9px'>{h}</th>" for h in th)
        + "</tr></thead><tbody>" + "".join(trs) + "</tbody></table>"
    )
    # ---- per-metric boxplots (distribution across patients, one panel per metric) ----
    fig = make_subplots(rows=2, cols=3, subplot_titles=[METRIC_LABEL[m] for m in metrics],
                        vertical_spacing=0.13, horizontal_spacing=0.07)
    for i, mt in enumerate(metrics):
        rr, cc = divmod(i, 3)
        for key in order:                       # consistent method order across panels
            vals = long[long["method"] == key][mt].dropna()
            fig.add_trace(go.Box(y=vals, name=key, boxpoints="outliers",
                                 marker=dict(size=3), line=dict(width=1.2),
                                 showlegend=False), row=rr + 1, col=cc + 1)
    fig.update_layout(template="plotly_white", height=820,
                      title="Per-metric distribution across the cohort (one box = one method; "
                            "ranked left→right by median variance ↓)")
    fig.update_xaxes(tickangle=-45)
    charts = fig.to_html(full_html=False, include_plotlyjs="cdn")

    out_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Cohort smoother comparison</title></head><body style='margin:24px'>"
        + table + "<div style='margin-top:30px'>" + charts + "</div></body></html>",
        encoding="utf-8")


def main():
    print("--- COHORT SMOOTHER COMPARISON ---")
    long = collect()
    if long.empty:
        print("No smoothed CSVs found. Run the Stage-2 smoothers first.")
        return
    summ = summarize(long)

    res_dir = Path("data/results"); res_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path("data/plots"); plots_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(res_dir / "cohort_comparison_per_patient.csv", index=False)
    summ.to_csv(res_dir / "cohort_comparison_summary.csv", index=False)
    out_html = plots_dir / "cohort_comparison.html"
    build_html(long, summ, out_html)

    print("\n=== per-method medians (ranked by variance ↓) ===")
    show = summ.sort_values("var_red_med", ascending=False)
    for _, r in show.iterrows():
        print(f"  {r['method']:9s}  var↓={r['var_red_med']:5.1f}%  rough↓={r['rough_red_med']:5.1f}%  "
              f"peak={r['peak_med']:.2f}  corr={r['corr_med']:.2f}  resid_std={r['resid_std_med']:5.2f}ms  "
              f"(n={int(r['n_patients'])})")
    print(f"\nWrote:\n  {res_dir/'cohort_comparison_per_patient.csv'}"
          f"\n  {res_dir/'cohort_comparison_summary.csv'}\n  {out_html}")


if __name__ == "__main__":
    main()
