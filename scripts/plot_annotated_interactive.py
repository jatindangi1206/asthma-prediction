"""Interactive (plotly) view of the annotated Dual-Track output for a patient.

Top panel : raw HRV (Track B) + smoothed baseline (Track A).
Bottom    : residual_hrv with the ensemble CPD anomalies, coloured by phenotype.
Causal, full series, range-slider; lines break at >=180-min gaps.

Usage (from asthma-prediction/):
    python scripts/plot_annotated_interactive.py <method> <pid> [<pid> ...]
    e.g. python scripts/plot_annotated_interactive.py rspf a32 a001
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PHENO_COLORS = {
    "KCPD_Distribution_Shift": "#9467bd",
    "HMM_State_Transition":          "#d62728",
    "Kalman_Gradual_Drift":          "#ff7f0e",
    "Kalman_Volatility_Spike":       "#e6194b",
    "Kalman_Autocorrelation_Change": "#bcbd22",
    "BOCPD_Mean_Shift":              "#17becf",
    "BOCPD_Variance_Change":         "#1f77b4",
}


def build(method, pid):
    f = Path(f"data/annotated_{method}/{pid}_{method}_annotated.csv")
    df = pd.read_csv(f)
    df["createdTime"] = pd.to_datetime(df["createdTime"])
    obs = df[df["gap_flag"] == 0]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.5, 0.5],
                        subplot_titles=("Track A vs Track B — raw HRV + smoothed baseline",
                                        "residual_hrv + ensemble CPD anomalies (coloured by phenotype)"))

    # top: raw + smoothed
    fig.add_trace(go.Scattergl(x=obs["createdTime"], y=obs["hrvValue"], mode="markers",
                               name="raw HRV", marker=dict(size=3, color="#b0b7c3", opacity=0.5)), row=1, col=1)
    fig.add_trace(go.Scattergl(x=df["createdTime"], y=df["smoothed_hrv"], mode="lines",
                               name="smoothed (Track A)", line=dict(color="#1f77b4", width=1.2),
                               connectgaps=False), row=1, col=1)

    # bottom: residual + anomalies
    fig.add_trace(go.Scattergl(x=df["createdTime"], y=df["residual_hrv"], mode="lines",
                               name="residual_hrv", line=dict(color="#2ca02c", width=0.8),
                               connectgaps=False), row=2, col=1)
    fired = df[df["anomaly_present"] == 1]
    for ph, c in PHENO_COLORS.items():
        s = fired[fired["primary_phenotype"] == ph]
        if len(s):
            fig.add_trace(go.Scattergl(
                x=s["createdTime"], y=s["residual_hrv"], mode="markers",
                name=f"{ph} ({len(s)})",
                marker=dict(size=7, color=c, line=dict(width=0.4, color="#333")),
                hovertemplate="%{x}<br>residual=%{y:.1f}<br>mag=%{customdata:.3f}<extra></extra>",
                customdata=s["magnitude"]), row=2, col=1)

    n_chunks = int(obs["chunk_id"].nunique())
    n_anom = int(df["anomaly_present"].sum())
    fig.update_layout(
        title=f"Patient {pid} — Dual-Track Residual + CPD ensemble (Track A = {method.upper()}). "
              f"{n_chunks} chunks, {n_anom} ensemble anomaly rows. Drag to zoom.",
        template="plotly_white", hovermode="x unified", height=760,
        legend=dict(orientation="h", y=1.07, x=0, font=dict(size=9)))
    fig.update_yaxes(title_text="HRV (ms)", row=1, col=1)
    fig.update_yaxes(title_text="residual (ms)", row=2, col=1)
    fig.add_hline(y=0, line=dict(color="#888", width=0.6), row=2, col=1)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.04), row=2, col=1)

    out = Path("data/plots") / f"{pid}_{method}_annotated_interactive.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"{pid}: {out}")


if __name__ == "__main__":
    method = sys.argv[1]
    for pid in sys.argv[2:]:
        build(method, pid)
