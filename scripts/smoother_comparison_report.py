"""Smoother comparison framework — side-by-side visual evaluation on one patient.

Runs every registered smoothing method on exactly the same raw HRV series for a
single (configurable) patient, computes comparison metrics, and writes ONE
interactive Plotly HTML report: a summary table at the top, then a synchronized
stack of raw-vs-smoothed charts (one per method).

MODULAR DESIGN
--------------
To add a newly implemented smoother to the report, add ONE entry to REGISTRY
below. Each entry names the module (in src/), its `smooth_dataframe` API, a
config instance, and a short parameter string. Nothing else needs to change.

Usage (from asthma-prediction/):
    python scripts/smoother_comparison_report.py [patient_id]   # default 0010
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
PROCESSED = Path("data/processed")
OUT_DIR = Path("data/plots")

# ==============================================================================
# SMOOTHER REGISTRY  — add a new method here and it appears in the report.
#   key      : short id
#   name     : display name
#   module   : module in src/ exposing smooth_dataframe(...) + a Config class
#   config   : "(ConfigClass, dict-of-overrides)"  (None -> module defaults)
#   params   : short human-readable parameter string (shown in the report)
#   enabled  : include in the run
# ==============================================================================
REGISTRY = [
    dict(key="rspf",      name="RS-PF (Regime-Switching Particle Filter)",
         module="rspf_smoother",   cfg="RSPFConfig",     params="K=3 regimes, N=500 particles", enabled=True),
    dict(key="krlst",     name="KRLS-T (Kernel RLS Tracker)",
         module="krlst_smoother",  cfg="KRLSTConfig",    params="lengthscale=60min, c=5, λ=0.995, budget=100", enabled=True),
    dict(key="gpssm",     name="State-Space GP (Markovian GP)",
         module="gpssm_smoother",  cfg="GPSSMConfig",    params="q_scale=0.05, r_scale=5.0 (GP-learned F)", enabled=True),
    dict(key="ossa",      name="OSSA (Online Singular Spectrum Analysis)",
         module="ossa_smoother",   cfg="OSSAConfig",     params="window=144, n_components=2", enabled=True),
    dict(key="kim",       name="Kim Filter (Markov-switching SSM)",
         module="kim_smoother",    cfg="KimConfig",      params="2 regimes, q_scale=0.1, r_scale=3.0", enabled=True),
    dict(key="gammadglm", name="Gamma CD-DGLM",
         module="gamma_dglm",      cfg="GammaDGLMConfig",params="deltrend=0.99, delseas=0.99, shape data-driven", enabled=True),
    dict(key="logmhw",    name="Log-MHW (Log Multiplicative Holt-Winters)",
         module="logmhw_smoother", cfg="LogMHWConfig",   params="α=0.08, γ=0.02 (statsmodels-optimised α/β/φ)", enabled=True),
    dict(key="hrkf",      name="H-RKF (Huber Robust Kalman, log-domain)",
         module="hrkf_smoother",   cfg="HRKFConfig",     params="deltrend=0.99, huber_δ=1.345", enabled=True),
    dict(key="anrewma",   name="AN-REWMA (Adaptive Norm + Robust EWMA)",
         module="anrewma_smoother",cfg="ANREWMAConfig",  params="norm_α=0.05, IMQ c=1.0", enabled=True),
    dict(key="ctimm",     name="CT-IMM (Continuous-Time Interacting Multiple Model)",
         module="ct_imm_smoother", cfg="CTIMMConfig",     params="3 modes (calm/normal/volatile), q_scale=0.001, sticky=0.95", enabled=True),
    # Original SMC particle filter (FFBS) — NON-CAUSAL and slow. Different API, so
    # it is read from its precomputed output rather than re-run inside the report.
    dict(key="pf",        name="Particle Filter (SMC + FFBS) — non-causal, offline",
         source="precomputed", file_tmpl="data/smoothed/{pid}_smoothed.csv",
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
    mod = __import__(entry["module"])
    Cfg = getattr(mod, entry["cfg"])
    cfg = Cfg()
    t0 = time.perf_counter()
    out = mod.smooth_dataframe(obs.copy(), patient_col="patient_id",
                               timestamp_col="createdTime", value_col="hrvValue",
                               config=cfg, out_col="smoothed_hrv")
    runtime = time.perf_counter() - t0
    out = out.sort_values("createdTime").reset_index(drop=True)
    n_chunks = int(out["chunk_id"].nunique()) if "chunk_id" in out else np.nan
    return out["smoothed_hrv"].to_numpy(), out["createdTime"], runtime, n_chunks


# ==============================================================================
# METRICS
# ==============================================================================
def peak_preservation(raw, sm, win=6):
    """Fraction of each raw peak's local excursion retained in the smoothed signal."""
    pk, _ = find_peaks(raw, prominence=0.5 * np.nanstd(raw))
    if pk.size == 0:
        return np.nan
    ratios = []
    n = len(raw)
    for p in pk:
        a, b = max(0, p - win), min(n, p + win + 1)
        raw_exc = raw[p] - np.min(raw[a:b])
        sm_exc = sm[p] - np.min(sm[a:b])
        if raw_exc > 1e-9:
            ratios.append(np.clip(sm_exc / raw_exc, 0, 1))
    return float(np.median(ratios)) if ratios else np.nan


def metrics(raw_full, sm_full):
    """Compute comparison metrics on rows where both raw and smoothed exist."""
    m = np.isfinite(raw_full) & np.isfinite(sm_full)
    raw, sm = raw_full[m], sm_full[m]
    if raw.size < 3:
        return None
    var_red = 100.0 * (1 - np.var(sm) / np.var(raw)) if np.var(raw) > 0 else np.nan
    rough_raw = np.mean(np.abs(np.diff(raw)))
    rough_red = 100.0 * (1 - np.mean(np.abs(np.diff(sm))) / rough_raw) if rough_raw > 0 else np.nan
    mean_err = abs(np.mean(sm) - np.mean(raw))
    mean_err_pct = 100.0 * mean_err / abs(np.mean(raw)) if np.mean(raw) != 0 else np.nan
    corr = float(np.corrcoef(raw, sm)[0, 1]) if raw.size > 1 else np.nan
    return dict(var_red=var_red, rough_red=rough_red, mean_err=mean_err,
                mean_err_pct=mean_err_pct, peak=peak_preservation(raw, sm),
                corr=corr, n_used=int(raw.size))


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
            print(f"  {e['key']:9s} ok  {runtime:5.1f}s  var_red={mt['var_red']:.0f}%  "
                  f"rough_red={mt['rough_red']:.0f}%  peak={mt['peak']:.2f}")
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
            fmt(r['runtime'], "{:.2f}"), fmt(r['var_red']), fmt(r['rough_red']),
            fmt(r['mean_err_pct']), fmt(r['peak'], "{:.2f}"), fmt(r['corr'], "{:.2f}"),
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
        "It is written as the <code>residual_hrv</code> column in each <code>data/smoothed_&lt;method&gt;/</code> "
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
