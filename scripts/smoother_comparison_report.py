"""Smoother comparison framework — side-by-side visual evaluation on one patient.

Runs every registered smoothing method on exactly the same raw HRV series for a
single (configurable) patient, computes comparison metrics, and writes ONE
interactive Plotly HTML report: a summary table at the top, then a synchronized
stack of raw-vs-smoothed charts (one per method).

MODULAR DESIGN
--------------
To add a newly implemented smoother to the report, add ONE entry to REGISTRY
below, derived from wearable.smoothing.REGISTRY — its `smooth_dataframe` API, a
config instance, and a short parameter string. Nothing else needs to change.

Usage (from wearable-prediction/):
    python scripts/smoother_comparison_report.py [patient_id]   # default 0010
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wearable import metrics as metric_pool                      # noqa: E402
from wearable import smoothing                                   # noqa: E402
from wearable.config import PROCESSED_DIR, smoothed_dir          # noqa: E402

PROCESSED = PROCESSED_DIR
OUT_DIR = Path("data/reports/smoother_comparison")

# Built from the one smoother registry — a method registered in
# wearable/smoothing/__init__.py appears in this report with no edit here. The
# parameter string shown in the report is just that method's registry defaults.
REGISTRY = [
    dict(key=key, name=f"{spec.label} ({key})",
         params=", ".join(f"{k}={v}" for k, v in spec.defaults.items()), enabled=True)
    for key, spec in smoothing.all_methods().items()
] + [
    # The original SMC particle filter (FFBS) is NON-CAUSAL and slow, and has a
    # different API, so it is read from its precomputed output rather than re-run.
    dict(key="pf", name="Particle Filter (SMC + FFBS) — non-causal, offline",
         source="precomputed", file_tmpl=f"{smoothed_dir('pf')}/{{pid}}_smoothed.csv",
         params="N=250 particles, M=50 FFBS paths (precomputed)", enabled=True),
]


# ==============================================================================
# DATA
# ==============================================================================
def load_patient(pid):
    df = pd.read_csv(PROCESSED / f"{pid}_processed.csv", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["createdTime"] = pd.to_datetime(df["createdTime"])
    df = df.sort_values("createdTime").reset_index(drop=True)
    n_total = len(df)
    obs = df[df["hrvValue"].notna()].copy()
    obs["patient_id"] = pid
    return df, obs, n_total


def run_smoother(entry, obs):
    """Run one registered smoother on the observed rows; return (smoothed_series, info)."""
    # Go through the registry so the report runs the method with exactly the pipeline's
    # configured knobs, not a bare Config().
    spec = smoothing.method(entry["key"])
    t0 = time.perf_counter()
    out = spec.entry()(obs.copy(), patient_col="patient_id",
                           timestamp_col="createdTime", value_col="hrvValue",
                           config=spec.config(), out_col="smoothed_hrv")
    runtime = time.perf_counter() - t0
    out = out.sort_values("createdTime").reset_index(drop=True)
    n_chunks = int(out["chunk_id"].nunique()) if "chunk_id" in out else np.nan
    return out["smoothed_hrv"].to_numpy(), out["createdTime"], runtime, n_chunks


# ==============================================================================
# METRICS
# ==============================================================================
def metrics(raw_full, sm_full):
    """Every signal metric in wearable/metrics/, keyed by metric name.

    ponytail: the formulas used to live here as well as in the pool. One definition —
    add a metric file and this report gains a column with no edit.
    """
    out = {}
    for name in metric_pool.names("signal"):
        try:
            out[name] = float(metric_pool.metric(name).load().compute(raw_full, sm_full))
        except Exception:
            out[name] = float("nan")
    m = np.isfinite(raw_full) & np.isfinite(sm_full)
    out["n_used"] = int(m.sum())
    return out if out["n_used"] >= 3 else None


# ==============================================================================
# REPORT
# ==============================================================================
def build_report(pid):
    df, obs, n_total = load_patient(pid)
    raw_full = pd.to_numeric(df["hrvValue"], errors="coerce").to_numpy()
    n_obs = int(np.isfinite(raw_full).sum())
    n_missing = n_total - n_obs

    rows = []          # metric rows for the table
    series = []        # (name, smoothed aligned to df rows) for charts
    for e in REGISTRY:
        if not e.get("enabled", True):
            continue
        try:
            if e.get("source") == "precomputed":
                # read an already-generated smoothed CSV (e.g. the slow non-causal PF)
                pf = pd.read_csv(e["file_tmpl"].format(pid=pid))
                pf["createdTime"] = pd.to_datetime(pf["createdTime"])
                sm_map = pd.Series(pf["smoothed_hrv"].to_numpy(), index=pf["createdTime"])
                runtime = np.nan                       # not re-run here
                ot = df.loc[np.isfinite(raw_full), "createdTime"]
                gaps = ot.diff().dt.total_seconds() / 60.0
                n_chunks = int((gaps >= 180).sum()) + 1
            else:
                sm_obs, t_obs, runtime, n_chunks = run_smoother(e, obs)
                sm_map = pd.Series(sm_obs, index=t_obs)
            # align smoothed (indexed by createdTime) back to the full df grid
            sm_full = df["createdTime"].map(sm_map).to_numpy(dtype=float)
            mt = metrics(raw_full, sm_full)
            n_after = int(np.isfinite(sm_full).sum())
            rows.append(dict(name=e["name"], params=e["params"], runtime=runtime,
                             n_chunks=n_chunks, n_after=n_after, **(mt or {})))
            series.append((e["name"], e["params"], runtime, sm_full, mt, n_after))
            print(f"  {e['key']:9s} ok  {runtime:5.1f}s  var_red={mt['variance_reduction']:.0f}%  "
                  f"rough_red={mt['roughness_reduction']:.0f}%  peak={mt['peak_preservation']:.2f}")
        except Exception as ex:
            print(f"  {e['key']:9s} FAILED: {ex}")
            rows.append(dict(name=e["name"], params=e["params"], runtime=np.nan,
                             error=str(ex)))

    # ---- summary table (HTML) -------------------------------------------------
    def fmt(v, s="{:.1f}"):
        return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else s.format(v)
    th = ("Method", "Runtime (s)", "Variance ↓ %", "Roughness ↓ %", "Mean err %",
          "Peak score", "Corr", "Chunks", "Obs→Smoothed")
    trs = []
    for r in rows:
        if "error" in r:
            trs.append(f"<tr><td>{r['name']}</td><td colspan='8' style='color:#b00'>FAILED: {r['error']}</td></tr>")
            continue
        trs.append("<tr>" + "".join(f"<td>{c}</td>" for c in [
            f"<b>{r['name']}</b><br><span style='color:#777;font-size:11px'>{r['params']}</span>",
            fmt(r['runtime'], "{:.2f}"), fmt(r['variance_reduction']), fmt(r['roughness_reduction']),
            fmt(r['mean_error_pct']), fmt(r['peak_preservation'], "{:.2f}"), fmt(r['correlation'], "{:.2f}"),
            fmt(r.get('n_chunks'), "{:.0f}"), f"{n_obs}→{r['n_after']}"]) + "</tr>")
    defs_html = (
        "<h3 style='font-family:Arial;margin-top:26px'>What each column means</h3>"
        "<table border='1' cellspacing='0' cellpadding='6' "
        "style='border-collapse:collapse;font-family:Arial;font-size:12.5px;max-width:1100px'>"
        "<thead style='background:#eef2f8'><tr><th>Column</th><th>Definition / formula</th>"
        "<th>What it measures</th></tr></thead><tbody>"
        "<tr><td><b>Runtime (s)</b></td><td>Wall-clock time to smooth this patient's observed series.</td>"
        "<td>Compute cost per patient.</td></tr>"
        "<tr><td><b>Variance ↓ %</b></td><td>100 · (1 − Var(smoothed) / Var(raw)), over rows where both exist.</td>"
        "<td>Share of the raw signal's variance that is no longer present in the smoothed output.</td></tr>"
        "<tr><td><b>Roughness ↓ %</b></td><td>100 · (1 − mean|Δsmoothed| / mean|Δraw|), where Δ is the "
        "difference between consecutive observed samples.</td>"
        "<td>Change in point-to-point movement (first-difference size) from raw to smoothed.</td></tr>"
        "<tr><td><b>Mean err %</b></td><td>100 · |mean(smoothed) − mean(raw)| / |mean(raw)|.</td>"
        "<td>Distance between the average level of the smoothed output and the average level of the raw signal.</td></tr>"
        "<tr><td><b>Peak score</b></td><td>Raw local maxima are found (prominence ≥ 0.5·std). For each, the height "
        "above the surrounding ±6-sample minimum is taken in raw and in smoothed; score = median of "
        "clip(smoothed-height / raw-height, 0, 1). Range 0–1.</td>"
        "<td>Fraction of each raw peak's local height that remains in the smoothed output.</td></tr>"
        "<tr><td><b>Corr</b></td><td>Pearson correlation between raw and smoothed, over rows where both exist.</td>"
        "<td>Linear agreement between the smoothed output and the raw signal.</td></tr>"
        "<tr><td><b>Chunks</b></td><td>Number of continuous segments after splitting at ≥180-min gaps; "
        "each is smoothed independently with the state reset at the boundary.</td>"
        "<td>How the series was segmented for the 180-minute missingness rule.</td></tr>"
        "<tr><td><b>Obs→Smoothed</b></td><td>Count of observed raw values → count of rows that received a smoothed "
        "value (segments shorter than 10 observed rows are left unsmoothed).</td>"
        "<td>Coverage of the output. The intro line also reports missing/gap rows the smoother skips.</td></tr>"
        "</tbody></table>"
        "<h3 style='font-family:Arial;margin-top:22px'>The residual (and where it comes from)</h3>"
        "<p style='font-family:Arial;font-size:12.5px;max-width:1100px;color:#333'>"
        "Each smoother is <b>Track A</b> — an estimate of the patient's expected HRV baseline "
        "(<code>smoothed_hrv</code>). The raw signal is <b>Track B</b>. Their difference, computed on every "
        "observed row, is the <b>residual</b>: &nbsp;"
        "<code>residual_hrv = hrvValue − smoothed_hrv</code>. "
        "It is written as the <code>residual_hrv</code> column in each <code>the smooth artifact's table</code> "
        "file and is the signal the change-point detectors operate on downstream. It is not a column in the "
        "table above; rather, the columns above (variance ↓, roughness ↓, peak score, corr) describe what Track A "
        "removes versus keeps — which is exactly what determines the contents of the residual.</p>")

    table = (f"<h2>Patient {pid} — Smoother Comparison</h2>"
             f"<p style='color:#555'>Raw observations: <b>{n_obs}</b> &nbsp;|&nbsp; "
             f"missing/gap rows handled: <b>{n_missing}</b> &nbsp;|&nbsp; "
             f"methods: <b>{len(series)}</b> &nbsp;|&nbsp; "
             f"metrics computed on observed rows. ↓ = reduction (higher = smoother). "
             f"Peak score &amp; Corr near 1 = better signal fidelity.</p>"
             "<table border='1' cellspacing='0' cellpadding='6' "
             "style='border-collapse:collapse;font-family:Arial;font-size:13px'>"
             "<thead style='background:#1f3864;color:#fff'><tr>"
             + "".join(f"<th>{h}</th>" for h in th) + "</tr></thead><tbody>"
             + "".join(trs) + "</tbody></table>")

    # ---- synchronized charts (raw on top, one row per method) ----------------
    obs_mask = np.isfinite(raw_full)
    titles = ["Raw HRV signal"] + [f"{nm} — {pr} | {rt:.2f}s" for nm, pr, rt, *_ in series]
    fig = make_subplots(rows=len(series) + 1, cols=1, shared_xaxes=True,
                        vertical_spacing=0.012, subplot_titles=titles)
    t = df["createdTime"]
    fig.add_trace(go.Scattergl(x=t[obs_mask], y=raw_full[obs_mask], mode="markers",
                               name="raw HRV", marker=dict(size=2.5, color="#9aa0aa", opacity=0.5),
                               showlegend=False), row=1, col=1)
    for i, (nm, pr, rt, sm_full, mt, n_after) in enumerate(series, start=2):
        fig.add_trace(go.Scattergl(x=t[obs_mask], y=raw_full[obs_mask], mode="markers",
                                   marker=dict(size=2, color="#cfd3da", opacity=0.4),
                                   showlegend=False, hoverinfo="skip"), row=i, col=1)
        fig.add_trace(go.Scattergl(x=t, y=sm_full, mode="lines",
                                   line=dict(color="#1f77b4", width=1.1), connectgaps=False,
                                   showlegend=False, name=nm), row=i, col=1)
    fig.update_layout(template="plotly_white", height=260 * (len(series) + 1),
                      title=f"Patient {pid} — raw vs smoothed (synchronized x-axis; drag/scroll to zoom)",
                      margin=dict(t=70, l=60, r=30, b=40))
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.02), row=len(series) + 1, col=1)
    for r in range(1, len(series) + 2):
        fig.update_yaxes(title_text="HRV (ms)", row=r, col=1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"smoother_comparison_{pid}.html"
    html = ("<html><head><meta charset='utf-8'><title>Smoother comparison "
            f"{pid}</title></head><body style='font-family:Arial;margin:24px'>"
            + table + defs_html + "<hr style='margin:24px 0'>"
            # include_plotlyjs=True embeds plotly.js INTO the file -> fully
            # standalone & offline (double-click to open, no internet needed).
            + fig.to_html(full_html=False, include_plotlyjs=True) + "</body></html>")
    out.write_text(html)
    print(f"\nReport: {out}")
    return out


if __name__ == "__main__":
    pid = sys.argv[1] if len(sys.argv) > 1 else "0010"
    print(f"Building smoother comparison report for patient {pid} ...")
    build_report(pid)
