"""Plot one patient from any smooth or detect artifact.

    python scripts/plot_patient.py smooth:gammadglm 0010
    python scripts/plot_patient.py detect:bocpd+kcpd+kalman+hmm a001 a004

Produces, per patient, in data/reports/plots/:
    <pid>_<method>.png               static 2-panel diagnostic (full series + densest 3 days)
    <pid>_<method>_interactive.html  plotly, zoomable, with a range slider

This replaces the eight near-identical `plot_<method>.py` scripts and
`plot_annotated_interactive.py`: the method is an argument and the data comes from the
artifact store, so a newly added smoother is plottable with no new file.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wearable import store                                        # noqa: E402
from wearable.__main__ import _resolve                            # noqa: E402

OUT_DIR = Path("data/reports/plots")
COLOR = "#ff7f0e"
PHENOTYPE_COLORS = {                      # CPD markers, by firing detector family
    "BOCPD": "#d62728", "KCPD": "#9467bd", "Kalman": "#2ca02c", "HMM": "#1f77b4",
}


def _slug(pid: str, label: str) -> str:
    """<pid>__<step>__<method>, lowercase snake_case — no spaces, parens or plus signs."""
    clean = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return f"{pid}__{clean}"


def _phenotype_color(name: str) -> str:
    for prefix, c in PHENOTYPE_COLORS.items():
        if str(name).startswith(prefix):
            return c
    return "#7f7f7f"


def _densest_window(obs: pd.DataFrame, col: str, span=pd.Timedelta(days=3)):
    """The 3-day window holding the most observations — the useful place to zoom."""
    if obs.empty:
        return None
    counts = obs.set_index("createdTime")[col].rolling(span).count()
    end = counts.reset_index(drop=True).idxmax()
    return obs["createdTime"].iloc[end] - span


def plot_patient(df: pd.DataFrame, pid: str, label: str, value_col: str,
                 baseline_col: str | None, events: pd.DataFrame | None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    obs = df[df[value_col].notna()]
    start = _densest_window(obs, value_col)
    zwin = df if start is None else df[(df["createdTime"] >= start) &
                                      (df["createdTime"] <= start + pd.Timedelta(days=3))]
    zobs = zwin[zwin[value_col].notna()]
    n_chunks = obs["chunk_id"].nunique() if "chunk_id" in obs else 0

    # ---- static -----------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9))
    for ax, frame, fobs, title, fmt in (
            (ax1, df, obs, f"Patient {pid} — {label} (full series, {n_chunks} chunks; "
                           f"line breaks at ≥180-min gaps)", "%Y-%m-%d"),
            (ax2, zwin, zobs, "Zoom: densest 3-day window", "%m-%d %H:%M")):
        ax.scatter(fobs["createdTime"], fobs[value_col], s=10, c="#9aa0aa",
                   alpha=0.55, label=value_col, zorder=1)
        if baseline_col and baseline_col in frame:
            ax.plot(frame["createdTime"], frame[baseline_col], c=COLOR, lw=1.1,
                    label=baseline_col, zorder=3)
        if events is not None and len(events):
            ev = events[(events["createdTime"] >= frame["createdTime"].min()) &
                        (events["createdTime"] <= frame["createdTime"].max())]
            for _, r in ev.iterrows():
                ax.axvline(r["createdTime"], color=_phenotype_color(r["primary_phenotype"]),
                           alpha=0.35, lw=0.9, zorder=2)
        ax.set_title(title)
        ax.set_ylabel(value_col)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    ax2.set_xlabel("time")
    fig.tight_layout()
    png = OUT_DIR / f"{_slug(pid, label)}.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"  static      -> {png}")

    # ---- interactive ------------------------------------------------------
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=obs["createdTime"], y=obs[value_col], mode="markers",
                               name=f"{value_col} (observed)",
                               marker=dict(size=3, color="#b0b7c3", opacity=0.55)))
    if baseline_col and baseline_col in df:
        fig.add_trace(go.Scattergl(x=df["createdTime"], y=df[baseline_col], mode="lines",
                                   name=baseline_col, connectgaps=False,
                                   line=dict(color=COLOR, width=1.3)))
    if events is not None and len(events):
        fig.add_trace(go.Scattergl(
            x=events["createdTime"], y=events[value_col], mode="markers",
            name="CPD event", text=events["primary_phenotype"],
            hovertemplate="%{x}<br>%{text}<extra></extra>",
            marker=dict(size=9, symbol="x",
                        color=[_phenotype_color(p) for p in events["primary_phenotype"]])))
    fig.update_layout(title=f"Patient {pid} — {label} (drag to zoom). "
                            f"Line breaks at ≥180-min gaps.",
                      xaxis_title="time", yaxis_title=value_col,
                      template="plotly_white", hovermode="x unified",
                      legend=dict(orientation="h", y=1.06, x=0))
    fig.update_xaxes(rangeslider_visible=True)
    html = OUT_DIR / f"{_slug(pid, label)}_interactive.html"
    fig.write_html(html, include_plotlyjs="cdn")
    print(f"  interactive -> {html}")


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    spec, pids = argv[0], argv[1:]
    art = _resolve(spec)
    table = art.manifest["tables"][0]
    df = art.read_table(table)
    df["createdTime"] = pd.to_datetime(df["createdTime"])
    df["p_id"] = df["p_id"].astype(str)

    if art.step == "smooth":
        value_col, baseline_col = "hrvValue", "smoothed_hrv"
    else:
        value_col = art.manifest["params"].get("signal_col", "hrvValue")
        baseline_col = None

    for pid in pids:
        g = df[df["p_id"] == pid].sort_values("createdTime")
        if g.empty:
            print(f"  ! patient {pid} not in {spec} "
                  f"({df['p_id'].nunique()} patients present)")
            continue
        events = None
        if "anomaly_present" in g:
            events = g[g["anomaly_present"] == 1]
        print(f"patient {pid} ({len(g):,} rows)")
        plot_patient(g, pid, f"{art.step} {art.method}", value_col,
                     baseline_col, events)


if __name__ == "__main__":
    main(sys.argv[1:])
