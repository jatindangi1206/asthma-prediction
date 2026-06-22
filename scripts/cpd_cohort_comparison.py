"""Cohort-level change-point (Stage-3) comparison across all Track-A methods.

For every smoother whose residual was annotated (`data/annotated_<method>/`), this
aggregates the CPD ensemble behaviour across the cohort:
  - alarm rate  = anomaly rows / armed rows   (how trigger-happy the ensemble is)
  - ensemble + per-detector EVENT counts (contiguous runs, not rows)
  - primary-phenotype mix (which kind of change dominates)

Per-patient detector stats are read from `data/results/annotation_<method>_log.csv`
(fast); the phenotype mix is tallied from the annotated CSVs.

Outputs (from asthma-prediction/):
  data/results/cpd_cohort_summary.csv      per-method aggregates
  data/plots/cpd_cohort_comparison.html    ranked table + detector boxplots + phenotype bars

Run:
    python scripts/cpd_cohort_comparison.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

METHOD_KEYS = ["pf", "rspf", "krlst", "gpssm", "ossa", "kim",
               "gammadglm", "logmhw", "hrkf", "anrewma", "ctimm"]
DETECTORS = ["bocpd", "kcpd", "kalman", "hmm"]
PHENO_COLORS = {
    "BOCPD_Mean_Shift": "#17becf", "BOCPD_Variance_Change": "#1f77b4",
    "KCPD_Distribution_Shift": "#9467bd", "Kalman_Volatility_Spike": "#e6194b",
    "Kalman_Gradual_Drift": "#ff7f0e", "Kalman_Autocorrelation_Change": "#bcbd22",
    "HMM_State_Transition": "#d62728",
}


def collect():
    per_patient, pheno = [], {}     # pheno[method] = Counter-like dict of phenotype -> rows
    for key in METHOD_KEYS:
        log = Path(f"data/results/annotation_{key}_log.csv")
        adir = Path(f"data/annotated_{key}")
        if not log.exists():
            print(f"  {key:9s}  (no annotation log — skipped)")
            continue
        df = pd.read_csv(log)
        ok = df[df["status"] == "success"].copy()
        if ok.empty or "n_armed" not in ok.columns:
            print(f"  {key:9s}  (no successful annotations — skipped)")
            continue
        ok["alarm_rate"] = np.where(ok["n_armed"] > 0, ok["n_anomaly"] / ok["n_armed"], np.nan)
        for _, r in ok.iterrows():
            rec = {"method": key, "alarm_rate": r["alarm_rate"],
                   "ensemble_events": r.get("ensemble_events", np.nan),
                   "n_armed": r["n_armed"]}
            for d in DETECTORS:
                rec[f"{d}_events"] = r.get(f"{d}_events", np.nan)
            per_patient.append(rec)
        # phenotype mix from annotated CSVs (armed anomaly rows only)
        cnt = {}
        for f in sorted(adir.glob("*_annotated.csv")):
            try:
                p = pd.read_csv(f, usecols=["primary_phenotype"])["primary_phenotype"]
            except Exception:
                continue
            vc = p[~p.isin(["normal", "burn_in"])].value_counts()
            for k, v in vc.items():
                cnt[k] = cnt.get(k, 0) + int(v)
        pheno[key] = cnt
        print(f"  {key:9s}  {len(ok)} patients,  {sum(cnt.values()):>7d} anomaly rows")
    return pd.DataFrame(per_patient), pheno


def summarize(long):
    recs = []
    for key in METHOD_KEYS:
        sub = long[long["method"] == key]
        if sub.empty:
            continue
        rec = {"method": key, "n_patients": len(sub),
               "alarm_rate_med": float(sub["alarm_rate"].median()),
               "alarm_rate_iqr": float(sub["alarm_rate"].quantile(.75) - sub["alarm_rate"].quantile(.25)),
               "ensemble_events_med": float(sub["ensemble_events"].median()),
               "ensemble_events_tot": int(sub["ensemble_events"].sum())}
        for d in DETECTORS:
            rec[f"{d}_events_tot"] = int(sub[f"{d}_events"].sum())
        recs.append(rec)
    return pd.DataFrame(recs)


def build_html(long, summ, pheno, out_path):
    order = summ.sort_values("alarm_rate_med")["method"].tolist()  # calmest -> noisiest
    # ---- ranked table ----
    th = (["Method", "Patients", "Alarm rate median (IQR)", "Ensemble events median",
           "Ensemble events total"] + [f"{d} events (tot)" for d in DETECTORS])
    trs = []
    for key in order:
        r = summ[summ["method"] == key].iloc[0]
        cells = [f"<b>{key}</b>", f"{int(r['n_patients'])}",
                 f"{r['alarm_rate_med']*100:.1f}% <span style='color:#888'>({r['alarm_rate_iqr']*100:.1f})</span>",
                 f"{r['ensemble_events_med']:.0f}", f"{r['ensemble_events_tot']}"]
        cells += [f"{int(r[f'{d}_events_tot'])}" for d in DETECTORS]
        trs.append("<tr>" + "".join(f"<td style='padding:5px 9px'>{c}</td>" for c in cells) + "</tr>")
    table = (
        "<h2 style='font-family:Arial'>Cohort change-point comparison — "
        f"{long['method'].nunique()} Track-A methods × {long.groupby('method').size().max()} patients</h2>"
        "<p style='font-family:Arial;color:#555;max-width:1150px'>The same 4-detector CPD ensemble "
        "run on each smoother's <code>residual_hrv</code>. <b>Alarm rate</b> = anomaly rows / armed rows "
        "(median across patients, IQR in parentheses) — a noisier residual (under-smoothed Track A) drives "
        "more alarms. <b>Events</b> are contiguous anomaly runs, not rows (apples-to-apples; the HMM marks "
        "every row of a sustained regime). Rows ranked calmest → noisiest. <b>pf</b> residual is non-causal "
        "(offline reference).</p>"
        "<table border='1' cellspacing='0' cellpadding='0' style='border-collapse:collapse;"
        "font-family:Arial;font-size:13px'><thead style='background:#1f3864;color:#fff'><tr>"
        + "".join(f"<th style='padding:6px 9px'>{h}</th>" for h in th)
        + "</tr></thead><tbody>" + "".join(trs) + "</tbody></table>"
    )
    # ---- per-detector event-count boxplots ----
    fig = make_subplots(rows=1, cols=len(DETECTORS) + 1,
                        subplot_titles=["ensemble"] + DETECTORS, horizontal_spacing=0.04)
    for j, col in enumerate(["ensemble_events"] + [f"{d}_events" for d in DETECTORS]):
        for key in order:
            fig.add_trace(go.Box(y=long[long["method"] == key][col].dropna(), name=key,
                                 boxpoints="outliers", marker=dict(size=3), line=dict(width=1.1),
                                 showlegend=False), row=1, col=j + 1)
    fig.update_layout(template="plotly_white", height=430,
                      title="Per-patient event counts by detector (one box = one method)")
    fig.update_xaxes(tickangle=-45)
    box_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    # ---- phenotype mix (stacked %) ----
    phenos = list(PHENO_COLORS)
    pfig = go.Figure()
    for ph in phenos:
        ys = []
        for key in order:
            tot = sum(pheno.get(key, {}).values()) or 1
            ys.append(100.0 * pheno.get(key, {}).get(ph, 0) / tot)
        pfig.add_trace(go.Bar(name=ph, x=order, y=ys, marker_color=PHENO_COLORS[ph]))
    pfig.update_layout(barmode="stack", template="plotly_white", height=460,
                       title="Primary-phenotype mix per method (% of anomaly rows)",
                       yaxis_title="% of anomaly rows", legend=dict(font=dict(size=10)))
    pheno_html = pfig.to_html(full_html=False, include_plotlyjs=False)

    out_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Cohort CPD comparison</title></head>"
        "<body style='margin:24px'>" + table
        + "<div style='margin-top:28px'>" + box_html + "</div>"
        + "<div style='margin-top:18px'>" + pheno_html + "</div></body></html>",
        encoding="utf-8")


def main():
    print("--- COHORT CPD COMPARISON ---")
    long, pheno = collect()
    if long.empty:
        print("No annotation logs found. Run src/03_annotate.py <method> first.")
        return
    summ = summarize(long)
    res = Path("data/results"); res.mkdir(parents=True, exist_ok=True)
    plots = Path("data/plots"); plots.mkdir(parents=True, exist_ok=True)
    summ.to_csv(res / "cpd_cohort_summary.csv", index=False)
    out = plots / "cpd_cohort_comparison.html"
    build_html(long, summ, pheno, out)

    print("\n=== per-method (ranked calmest -> noisiest) ===")
    for _, r in summ.sort_values("alarm_rate_med").iterrows():
        print(f"  {r['method']:9s}  alarm={r['alarm_rate_med']*100:4.1f}%  "
              f"ens.events median={r['ensemble_events_med']:4.0f} total={r['ensemble_events_tot']:5d}  "
              f"| bocpd={r['bocpd_events_tot']:5d} kcpd={r['kcpd_events_tot']:5d} "
              f"kalman={r['kalman_events_tot']:5d} hmm={r['hmm_events_tot']:5d}")
    print(f"\nWrote:\n  {res/'cpd_cohort_summary.csv'}\n  {out}")


if __name__ == "__main__":
    main()
