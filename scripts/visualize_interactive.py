#!/usr/bin/env python3
"""scripts/visualize_interactive.py — interactive HTML overlay of smoothed HRV.

Runs from asthma-prediction/:
    python scripts/visualize_interactive.py            # all smoothed patients
    python scripts/visualize_interactive.py 0010       # one patient

Raw HRV (markers) + smoothed_hrv + true_trend_level (lines that break at gaps),
with zoom / pan / hover and an x-axis range slider. The static PNGs from
scripts/visualize.py are left untouched. Outputs → data/plots/<stem>_interactive.html
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

SMOOTHED_DIR = Path("./data/smoothed")
PLOTS_DIR = Path("./data/plots")


def build_figure(df: pd.DataFrame, stem: str) -> go.Figure:
    t = pd.to_datetime(df["createdTime"])
    fig = go.Figure()
    # Scattergl (WebGL) keeps pan/zoom smooth at ~20k+ points. connectgaps=False
    # so the smoothed lines break at NaN gap-fill rows instead of bridging gaps.
    fig.add_trace(go.Scattergl(x=t, y=df["hrvValue"], mode="markers", name="raw HRV",
                               marker=dict(size=3, color="lightcoral")))
    fig.add_trace(go.Scattergl(x=t, y=df["smoothed_hrv"], mode="lines", connectgaps=False,
                               name="smoothed (level+circadian)", line=dict(color="seagreen", width=1.5)))
    fig.add_trace(go.Scattergl(x=t, y=df["true_trend_level"], mode="lines", connectgaps=False,
                               name="trend level (baseline)", line=dict(color="navy", width=2)))
    fig.update_layout(
        title=f"Patient {stem} — smoothed HRV (interactive)",
        xaxis=dict(title="Time", rangeslider=dict(visible=True)),
        yaxis_title="HRV", hovermode="x unified", template="plotly_white",
    )
    return fig


def process(path: Path) -> None:
    stem = path.stem.replace("_smoothed", "")
    df = pd.read_csv(path, encoding="utf-8-sig")
    out = PLOTS_DIR / f"{stem}_interactive.html"
    build_figure(df, stem).write_html(out, include_plotlyjs="cdn")
    print(f"  ✓ {stem} → {out}")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    target = sys.argv[1] if len(sys.argv) > 1 else None
    pattern = f"*{target}*_smoothed.csv" if target else "*_smoothed.csv"
    files = sorted(SMOOTHED_DIR.glob(pattern))
    if not files:
        sys.exit(f"No smoothed CSVs matching '{pattern}' in {SMOOTHED_DIR.resolve()}")
    for f in files:
        process(f)


if __name__ == "__main__":
    main()
