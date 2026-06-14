"""Visualize the GP-SSM smoother output for one patient.

Produces:
  data/plots/<pid>_gpssm.png               -- static 2-panel diagnostic
  data/plots/<pid>_gpssm_interactive.html  -- interactive (zoom/pan) plotly

Usage (from asthma-prediction/):
    python scripts/plot_gpssm.py 0010
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go

PID = sys.argv[1] if len(sys.argv) > 1 else "0010"
df = pd.read_csv(Path(f"data/smoothed_gpssm/{PID}_gpssm.csv"))
df["createdTime"] = pd.to_datetime(df["createdTime"])
OUT_DIR = Path("data/plots"); OUT_DIR.mkdir(parents=True, exist_ok=True)
COL = "#2ca02c"   # GP-SSM green (distinct from KRLST blue)

obs = df[df["gap_flag"] == 0]
span = pd.Timedelta(days=3)
start = obs["createdTime"].iloc[
    obs.set_index("createdTime")["hrvValue"].rolling(span).count()
       .reset_index(drop=True).idxmax()] - span
zwin = df[(df["createdTime"] >= start) & (df["createdTime"] <= start + span)]
zobs = zwin[zwin["gap_flag"] == 0]

# ============================ STATIC (matplotlib) =========================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9))

ax1.plot(df["createdTime"], df["smoothed_hrv"], c=COL, lw=0.6, alpha=0.8,
         label="GP-SSM smoothed")
ax1.set_title(f"Patient {PID} — GP-SSM causal smoother (full series, "
              f"{obs['chunk_id'].nunique()} chunks; line breaks at ≥180-min gaps)")
ax1.set_ylabel("HRV (SDNN, ms)")
ax1.legend(loc="upper right", fontsize=8)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

ax2.scatter(zobs["createdTime"], zobs["hrvValue"], s=12, c="#9aa0aa", alpha=0.6,
            label="raw HRV", zorder=1)
ax2.plot(zwin["createdTime"], zwin["smoothed_hrv"], c=COL, lw=1.8,
         label="GP-SSM smoothed", zorder=3)
ax2.set_title("Zoom: densest 3-day window — GP-learned dynamics + causal Kalman filter")
ax2.set_ylabel("HRV (SDNN, ms)"); ax2.set_xlabel("time")
ax2.legend(loc="upper right", fontsize=8)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

for ax in (ax1, ax2):
    ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT_DIR / f"{PID}_gpssm.png", dpi=130)
plt.close(fig)
print(f"static  -> {OUT_DIR / f'{PID}_gpssm.png'}")

# ========================= INTERACTIVE (plotly) ===========================
fig = go.Figure()
fig.add_trace(go.Scattergl(
    x=obs["createdTime"], y=obs["hrvValue"], mode="markers", name="raw HRV (observed)",
    marker=dict(size=3, color="#b0b7c3", opacity=0.55)))
fig.add_trace(go.Scattergl(
    x=df["createdTime"], y=df["smoothed_hrv"], mode="lines", name="GP-SSM smoothed",
    line=dict(color=COL, width=1.3), connectgaps=False))
fig.update_layout(
    title=f"Patient {PID} — GP-SSM causal smoother (drag to zoom, use range slider). "
          f"Line breaks at ≥180-min gaps.",
    xaxis_title="time", yaxis_title="HRV (SDNN, ms)",
    template="plotly_white", hovermode="x unified",
    legend=dict(orientation="h", y=1.06, x=0))
fig.update_xaxes(rangeslider_visible=True)
fig.write_html(OUT_DIR / f"{PID}_gpssm_interactive.html", include_plotlyjs="cdn")
print(f"interactive -> {OUT_DIR / f'{PID}_gpssm_interactive.html'}")
