#!/usr/bin/env python3
"""Interactive report for ONE patient's final modeling CSVs (Stage 04 output).

Reads every  data/runs/<method>/modeling/<pid>__sm-<method>__model.csv  it can find
for the patient and renders a single self-contained HTML page: eight time-linked
panels with a method switcher, a checkbox-driven overlay panel, and a per-method
comparison table.

Run manually, one patient at a time:
    python scripts/plot_model_interactive.py 0010
    python scripts/plot_model_interactive.py 0010 --open
    python scripts/plot_model_interactive.py 0010 --methods gammadglm,rspf

Output: data/runs/_reports/<pid>_model.html  (opens offline — plotly.js is inlined)

Design notes (why it looks the way it does):
  * One measure per panel — never two y-scales on one plot. The overlay panel is the
    one place channels share an axis, and it only does so after z-scoring them to a
    common base, with the axis labelled in SD so nobody reads absolute values off it.
  * Colour carries role, not decoration: blue = the model's baseline, grey = raw
    observation, red = anomaly (status), violet = candidate_event. The four detector
    rows are labelled on the y-axis, so identity never rests on hue.
  * Lines BREAK at >=180-min voids (the same H8 threshold the pipeline segments on),
    so a wear gap never reads as a smooth interpolation.
  * Only observed rows are drawn; the 10-min grid is ~50% gap-filler.
  * Palette is the validated light-surface instance (CVD-checked); the sub-3:1 slots
    are the labelled detector rows, which satisfies the relief rule.
"""

import argparse
import json
import sys
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "runs"
OUT_DIR = RUNS / "_reports"
DIV_ID = "hrvchart"

# --- validated palette (light surface #fcfcfb) ------------------------------
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
VIOLET, CRITICAL, GREEN = "#4a3aa7", "#d03b3b", "#008300"
DETECTORS = [("bocpd", BLUE), ("kcpd", ORANGE), ("kalman", AQUA), ("hmm", YELLOW)]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# The overlay panel is the clinical adjudication view: put the annotated change points
# beside every vital at once, so a reviewer can judge whether a detected change is real.
# (col, label, colour, unit, method_dependent)
MAGENTA = "#e87ba4"
OVERLAY = [
    ("hrvValue",     "raw HRV",      MUTED,   "ms",   False),
    ("smoothed_hrv", "smoothed HRV", BLUE,    "ms",   True),
    ("residual_hrv", "residual",     GREEN,   "ms",   True),
    ("lastRate",     "heart rate",   ORANGE,  "bpm",  False),
    ("temperature",  "temperature",  AQUA,    "°F",   False),
    ("spo2Value",    "SpO₂",         VIOLET,  "%",    False),
    ("steps",        "steps",        YELLOW,  "",     False),
    ("sleep_stage",  "sleep stage",  MAGENTA, "code", False),
]
# Annotation layers for the overlay — full-height verticals, so they read as events rather
# than as another data series. Colours match panels 1-2 so the mapping is already learned.
OVERLAY_MARKS = [("changepoints", "change points", CRITICAL),
                 ("candidates", "candidate events", VIOLET)]
OV_Y = (-4.0, 8.0)          # fixed band: verticals need a known extent to span

PANELS = ("HRV — raw vs smoothed baseline", "Residual — CPD substrate & labels",
          "Which detector fired", "Heart rate", "Temperature", "Steps", "Sleep stage",
          "Overlay — annotations vs vitals (position is z-scored; tooltip shows real values)")


def find_models(pid, only=None):
    """{method: path} for every Stage-04 CSV belonging to this patient."""
    found = {}
    for p in sorted(RUNS.glob(f"*/modeling/{pid}__sm-*__model.csv")):
        method = p.parent.parent.name
        if only and method not in only:
            continue
        found[method] = p
    return found


def break_gaps(d, ycol, threshold_min=180):
    """(x, y) with a NaN inserted at every >=180-min void, so the line BREAKS there.

    Without this, two readings either side of a multi-day wear gap get joined by a
    straight diagonal — implying continuity the pipeline deliberately never invents.
    180 min is the same H8 threshold the smoothers and CPD segment on.
    """
    d = d[d[ycol].notna()]
    x, y = d["createdTime"].to_numpy(), d[ycol].to_numpy(dtype=float)
    if len(x) < 2:
        return x, y
    gaps = np.where(np.diff(x).astype("timedelta64[m]").astype(int) >= threshold_min)[0] + 1
    return np.insert(x, gaps, np.datetime64("NaT")), np.insert(y, gaps, np.nan)


def events(s):
    """Distinct events = contiguous runs of 1, not flagged rows."""
    return int(((s == 1) & (s.shift(1, fill_value=0) == 0)).sum())


def _ov_trace(d, col, label, colr, unit):
    """One overlay channel: plotted as a z-score, but hovering reports the REAL reading.

    Sharing one axis across ms / bpm / °F / counts is only defensible after indexing to a
    common base, so y is a z-score. A clinician adjudicating a change point needs the actual
    value though, so the real reading rides along in customdata and is what the tooltip shows.
    """
    v = d[col]
    sd = v.std()
    z = (v - v.mean()) / (sd if sd and np.isfinite(sd) and sd > 0 else 1.0)
    tmp = d.assign(_z=z, _raw=v)
    x, y = break_gaps(tmp, "_z")
    # break_gaps inserts NaN rows, so customdata has to be padded identically
    keep = tmp[tmp["_z"].notna()]
    raw = keep["_raw"].to_numpy(dtype=float)
    t = keep["createdTime"].to_numpy()
    gaps = (np.where(np.diff(t).astype("timedelta64[m]").astype(int) >= 180)[0] + 1
            if len(t) > 1 else np.array([], int))
    cd = np.insert(raw, gaps, np.nan)
    suffix = f" {unit}" if unit else ""
    return go.Scattergl(x=x, y=y, mode="lines", name=label, visible=False,
                        line=dict(width=1.2, color=colr), showlegend=False, customdata=cd,
                        hovertemplate=f"%{{customdata:.1f}}{suffix}<extra>{label}</extra>")


def _vlines(times, customdata=None):
    """Full-height vertical segments at `times` (x,y,customdata), NaN-separated per line.

    Drawn as verticals rather than markers so annotations read as events spanning every
    channel, which is the whole point of the overlay: line the change point up against the
    vitals underneath it.
    """
    lo, hi = OV_Y
    xs, ys, cds = [], [], []
    for i, t in enumerate(times):
        xs += [t, t, None]
        ys += [lo, hi, None]
        if customdata is not None:
            row = list(customdata[i])
            cds += [row, row, row]
    return xs, ys, (np.array(cds, dtype=object) if cds else None)


def build(models):
    fig = make_subplots(rows=8, cols=1, shared_xaxes=True, vertical_spacing=0.028,
                        row_heights=[.19, .17, .11, .09, .09, .09, .09, .17],
                        subplot_titles=PANELS)
    methods = list(models)
    first = "gammadglm" if "gammadglm" in methods else methods[0]
    meta = []      # one descriptor per trace; JS uses it to compute visibility

    def add(tr, row, kind="shared", method=None, col=None):
        fig.add_trace(tr, row=row, col=1)
        meta.append({"kind": kind, "method": method, "col": col})

    base = pd.read_csv(models[first], parse_dates=["createdTime"], dtype={"patient_id": str})

    # ---- shared, method-independent channels -------------------------------
    obs = base[base.hrvValue.notna()]
    add(go.Scattergl(x=obs.createdTime, y=obs.hrvValue, mode="markers", name="raw HRV",
                     marker=dict(size=2.5, color=MUTED, opacity=.55),
                     hovertemplate="%{y:.0f} ms<extra>raw</extra>"), 1)
    for col, row, label, unit in (("lastRate", 4, "heart rate", "bpm"),
                                  ("temperature", 5, "temperature", "°F"),
                                  ("steps", 6, "steps", "")):
        gx, gy = break_gaps(base, col)
        add(go.Scattergl(x=gx, y=gy, mode="lines", name=label, showlegend=False,
                         line=dict(width=1, color=MUTED),
                         hovertemplate=f"%{{y:.1f}} {unit}<extra>{label}</extra>"), row)
    sl = base[base.sleep_stage.notna()]
    add(go.Scattergl(x=sl.createdTime, y=sl.sleep_stage, mode="markers", name="sleep stage",
                     marker=dict(size=3, color=VIOLET, opacity=.6), showlegend=False,
                     hovertemplate="code %{y:.0f}<extra>sleep</extra>"), 7)

    # ---- overlay panel: shared z-scored axis, REAL values in the tooltip ----
    # The y position is a z-score (that is the only honest way to share one axis across
    # ms / bpm / °F / counts), but a clinician needs the actual reading — so every point
    # carries its real value in customdata and the tooltip shows that, never the z-score.
    for col, label, colr, unit, per_method in OVERLAY:
        if per_method:
            continue                                  # added inside the per-method loop below
        add(_ov_trace(base, col, label, colr, unit), 8, "ov", col=col)

    # ---- per-method traces --------------------------------------------------
    for m in methods:
        d = pd.read_csv(models[m], parse_dates=["createdTime"], dtype={"patient_id": str})
        vis = (m == first)
        gx, gy = break_gaps(d, "smoothed_hrv")
        add(go.Scattergl(x=gx, y=gy, mode="lines", name="smoothed baseline", visible=vis,
                         line=dict(width=1.2, color=BLUE),
                         hovertemplate="%{y:.1f} ms<extra>smoothed</extra>"), 1, "method", m)
        rx, ry = break_gaps(d, "residual_hrv")
        add(go.Scattergl(x=rx, y=ry, mode="lines", name="residual", visible=vis, opacity=.75,
                         line=dict(width=.8, color=MUTED),
                         hovertemplate="%{y:+.1f} ms<extra>residual</extra>"), 2, "method", m)
        # method-dependent overlay channels (visible only when ticked AND this method selected)
        for col, label, colr, unit, per_method in OVERLAY:
            if per_method:
                add(_ov_trace(d, col, label, colr, unit), 8, "ovm", m, col=col)
        # annotation layers: full-height verticals carrying the CPD detail in the tooltip
        onset = d[(d.anomaly_present == 1) & (d.anomaly_present.shift(1, fill_value=0) == 0)]
        cd = np.stack([onset.primary_phenotype.to_numpy(),
                       onset.bocpd_anomaly.to_numpy(), onset.kcpd_anomaly.to_numpy(),
                       onset.kalman_anomaly.to_numpy(), onset.hmm_anomaly.to_numpy()], axis=1)
        vx, vy, vc = _vlines(onset.createdTime.to_numpy(), cd)
        add(go.Scattergl(x=vx, y=vy, mode="lines", name="change point", visible=False,
                         line=dict(width=1, color=CRITICAL), opacity=.38,
                         showlegend=False, customdata=vc,
                         hovertemplate="<b>CHANGE POINT</b> %{customdata[0]}<br>"
                                       "bocpd %{customdata[1]} · kcpd %{customdata[2]} · "
                                       "kalman %{customdata[3]} · hmm %{customdata[4]}"
                                       "<extra></extra>"),
            8, "ovm", m, col="changepoints")
        cev = d[d.candidate_event == 1]
        kx, ky, _ = _vlines(cev.createdTime.to_numpy(), None)
        add(go.Scattergl(x=kx, y=ky, mode="lines", name="candidate", visible=False,
                         line=dict(width=1.4, color=VIOLET, dash="dot"), opacity=.85,
                         showlegend=False,
                         hovertemplate="<b>CANDIDATE</b> post-gap check<extra></extra>"),
            8, "ovm", m, col="candidates")
        a = d[(d.anomaly_present == 1) & d.residual_hrv.notna()]
        add(go.Scattergl(x=a.createdTime, y=a.residual_hrv, mode="markers", name="anomaly",
                         marker=dict(size=4.5, color=CRITICAL), visible=vis,
                         customdata=a.primary_phenotype,
                         hovertemplate="%{y:+.1f} ms<br>%{customdata}<extra>anomaly</extra>"),
            2, "method", m)
        c = d[d.candidate_event == 1]
        add(go.Scattergl(x=c.createdTime, y=c.residual_hrv.fillna(0), mode="markers",
                         name="candidate_event", visible=vis,
                         marker=dict(size=9, color=VIOLET, symbol="diamond-open",
                                     line=dict(width=1.6, color=VIOLET)),
                         hovertemplate="post-gap check<extra>candidate</extra>"), 2, "method", m)
        for i, (det, colr) in enumerate(DETECTORS):
            f = d[d[f"{det}_anomaly"] == 1]
            add(go.Scattergl(x=f.createdTime, y=[i] * len(f), mode="markers", name=det,
                             marker=dict(size=4, color=colr, symbol="line-ns-open",
                                         line=dict(width=1.4, color=colr)),
                             visible=vis, showlegend=False,
                             hovertemplate=f"{det} fired<extra></extra>"), 3, "method", m)

    order = [first] + [m for m in methods if m != first]

    # ---- chrome -------------------------------------------------------------
    fig.update_layout(
        height=1620, hovermode="x unified", template="plotly_white", autosize=True,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=INK_2),
        margin=dict(l=64, r=24, t=104, b=44),
        legend=dict(orientation="h", y=1.06, x=1, xanchor="right", yanchor="bottom",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRID,
                        font=dict(family=FONT, size=12, color=INK)))
    # type="date" is explicit because the overlay panel starts with every trace hidden;
    # with no visible data plotly infers a linear axis and the shared dates become 0,1,2…
    fig.update_xaxes(type="date", showgrid=False, showline=True, linecolor=AXIS,
                     ticks="outside", tickcolor=AXIS, tickfont=dict(color=MUTED, size=11))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False,
                     tickfont=dict(color=MUTED, size=11), title_font=dict(size=11, color=MUTED))
    for row, title in ((1, "ms"), (2, "ms"), (4, "bpm"), (5, "°F"), (6, "count"),
                       (7, "code"), (8, "SD")):
        fig.update_yaxes(title_text=title, row=row, col=1)
    fig.update_yaxes(row=3, col=1, tickmode="array", tickvals=list(range(len(DETECTORS))),
                     ticktext=[d for d, _ in DETECTORS], range=[-.6, len(DETECTORS) - .4],
                     showgrid=False)
    fig.update_yaxes(row=8, col=1, zeroline=True, zerolinecolor=GRID, zerolinewidth=1,
                     range=list(OV_Y), autorange=False)
    # Six months of 10-min data is an unreadable block at full extent, so open on the first
    # three weeks. The slider's OWN range must be pinned to the whole record: left implicit,
    # plotly derives it from the axis range and snaps the view straight back to full width.
    # autorange=False on BOTH the axis and the slider: the slider re-derives its extent from
    # visible data by default, so every overlay checkbox toggle would otherwise yank the view.
    t0, t1 = base.createdTime.min(), base.createdTime.max()
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=.03, bgcolor="#f2f1ec",
                                      autorange=False, range=[t0, t1]), row=8, col=1)
    fig.update_xaxes(autorange=False, range=[t0, t0 + pd.Timedelta(days=21)])
    for ann in fig.layout.annotations:
        ann.font.update(size=13, color=INK, family=FONT)
        ann.x, ann.xanchor = 0, "left"
    return fig, base, meta, order


def page(pid, fig, base, models, meta, order):
    span = f"{base.createdTime.min():%d %b %Y} → {base.createdTime.max():%d %b %Y}"
    n, nobs = len(base), int(base.hrvValue.notna().sum())
    tiles = [("patient", pid), ("rows", f"{n:,}"), ("HRV observed", f"{nobs:,}"),
             ("gap-filler", f"{n-nobs:,}"), ("methods", str(len(models)))]
    tiles_html = "".join(f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div></div>'
                         for k, v in tiles)
    boxes = "".join(
        f'<label class="cb"><input type="checkbox" data-col="{c}">'
        f'<span class="sw" style="background:{col}"></span>{lbl}</label>'
        for c, lbl, col, _u, _pm in OVERLAY)
    marks = "".join(
        f'<label class="cb mark"><input type="checkbox" data-col="{c}">'
        f'<span class="sw v" style="background:{col}"></span>{lbl}</label>'
        for c, lbl, col in OVERLAY_MARKS)
    picker = ('<select id="methodsel">'
              + "".join(f'<option value="{m}">{m}</option>' for m in order) + "</select>")

    rows = []
    for m, p in models.items():
        d = pd.read_csv(p, usecols=["anomaly_present", "candidate_event", "bocpd_anomaly",
                                    "kcpd_anomaly", "kalman_anomaly", "hmm_anomaly"])
        rows.append(f"<tr><td class='m'>{m}</td><td>{events(d.anomaly_present)}</td>"
                    f"<td>{events(d.bocpd_anomaly)}</td><td>{events(d.kcpd_anomaly)}</td>"
                    f"<td>{events(d.kalman_anomaly)}</td><td>{events(d.hmm_anomaly)}</td>"
                    f"<td>{int(d.candidate_event.sum())}</td></tr>")
    table = ("<table><thead><tr><th>method</th><th>ensemble events</th><th>bocpd</th><th>kcpd</th>"
             "<th>kalman</th><th>hmm</th><th>candidate</th></tr></thead><tbody>"
             + "".join(rows) + "</tbody></table>")

    meta_json = json.dumps(meta)
    body = fig.to_html(full_html=False, include_plotlyjs=True, div_id=DIV_ID,
                       # scrollZoom off on purpose: this page is ~1600px of stacked panels, and
                       # wheel-to-zoom traps the cursor so you can never scroll past the chart.
                       # Zoom stays available via the toolbar, drag-select, and the range slider.
                       config={"displaylogo": False, "scrollZoom": False, "responsive": True,
                               "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]})
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Patient {pid} — HRV model dataset</title><style>
  :root{{color-scheme:light}} *{{box-sizing:border-box}}
  body{{margin:0;background:#f9f9f7;color:{INK};font:14px/1.5 {FONT};
       -webkit-font-smoothing:antialiased}}
  .wrap{{max-width:1240px;margin:0 auto;padding:40px 24px 72px}}
  h1{{font-size:22px;font-weight:600;letter-spacing:-.01em;margin:0 0 4px}}
  .sub{{color:{MUTED};font-size:13px;margin:0 0 28px}}
  .tiles{{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:24px}}
  .tile{{background:{SURFACE};border:1px solid rgba(11,11,11,.10);border-radius:10px;
        padding:12px 18px;min-width:112px}}
  .tile .k{{color:{MUTED};font-size:11px;text-transform:uppercase;letter-spacing:.06em}}
  .tile .v{{color:{INK};font-size:20px;font-weight:600;margin-top:3px;
           font-variant-numeric:tabular-nums}}
  .card{{background:{SURFACE};border:1px solid rgba(11,11,11,.10);border-radius:12px;
        padding:12px 8px 4px}}
  .ovbar{{display:flex;flex-wrap:wrap;align-items:center;gap:8px;padding:12px 14px 0}}
  .ovbar .lab{{color:{MUTED};font-size:11px;text-transform:uppercase;letter-spacing:.06em;
              margin-right:4px}}
  .cb{{display:inline-flex;align-items:center;gap:7px;padding:6px 12px;cursor:pointer;
      border:1px solid rgba(11,11,11,.12);border-radius:999px;font-size:13px;
      background:#fff;user-select:none}}
  .cb:hover{{background:#f4f3ef}}
  .cb input{{margin:0;accent-color:{BLUE}}}
  #methodsel{{font:13px {FONT};color:{INK};background:#fff;border:1px solid rgba(11,11,11,.12);
             border-radius:8px;padding:6px 10px;cursor:pointer}}
  .sw{{width:11px;height:3px;border-radius:2px;display:inline-block}}
  .sw.v{{width:3px;height:11px}}
  .cb.mark{{border-style:dashed}}
  .cb.btn{{border-color:rgba(11,11,11,.18);color:{INK_2};font:12px {FONT};padding:6px 12px}}
  h2{{font-size:15px;font-weight:600;margin:36px 0 10px}}
  table{{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;
        background:{SURFACE};border:1px solid rgba(11,11,11,.10);border-radius:12px;
        overflow:hidden;width:100%}}
  th,td{{padding:9px 14px;text-align:right;border-bottom:1px solid {GRID}}}
  th{{color:{MUTED};font-weight:500;font-size:11px;text-transform:uppercase;letter-spacing:.05em}}
  th:first-child,td.m{{text-align:left}} td.m{{color:{INK};font-weight:500}}
  tbody tr:last-child td{{border-bottom:0}} tbody tr:hover{{background:#f4f3ef}}
  .note{{color:{MUTED};font-size:12px;margin-top:14px;max-width:74ch}}
</style></head><body><div class="wrap">
  <h1>Patient {pid} — HRV modeling dataset</h1>
  <p class="sub">{span} · Stage&nbsp;04 output · switch smoothing method with the selector above the chart</p>
  <div class="tiles">{tiles_html}</div>
  <div class="card">
    <div class="ovbar"><span class="lab">smoother</span>{picker}
      <span class="lab" style="margin-left:12px">annotations</span>{marks}</div>
    <div class="ovbar"><span class="lab">overlay channels</span>{boxes}
      <button id="ovall" class="cb btn">all</button>
      <button id="ovnone" class="cb btn">none</button></div>
    {body}
  </div>
  <h2>Detected events by method</h2>
  {table}
  <p class="note">Events are contiguous runs, not flagged rows. The ensemble is an OR of the four
  detectors, so it is recall-first. <b>candidate_event</b> marks the first reading after a gap of
  1–21 days — flagged for verification rather than silently absorbed into the baseline.
  Lines break at gaps of 3&nbsp;h or more, so a break is real missing wear, never interpolation.
  The overlay panel z-scores each channel (their units differ, so they are indexed to a common
  base) — read it for <i>shape and timing</i>, not absolute values.</p>
<script>
  // Visibility is a function of (selected smoother) x (ticked overlay boxes). Plotly's own
  // updatemenu writes an absolute visibility array, so it would clobber the checkboxes every
  // time you switched method — hence one state function owning both controls.
  var META = {meta_json};
  var GD = document.getElementById({DIV_ID!r});
  var SEL = document.getElementById('methodsel');

  function applyState() {{
    var method = SEL.value;
    var on = {{}};
    document.querySelectorAll('.ovbar input[type=checkbox]').forEach(function (cb) {{
      on[cb.dataset.col] = cb.checked;
    }});
    var vis = META.map(function (t) {{
      if (t.kind === 'method') return t.method === method;
      if (t.kind === 'ov')     return !!on[t.col];
      if (t.kind === 'ovm')    return !!on[t.col] && t.method === method;
      return true;                                   // shared channels
    }});
    Plotly.restyle(GD, {{visible: vis}}, vis.map(function (_, i) {{ return i; }}));
  }}
  function setAll(v) {{
    document.querySelectorAll('.ovbar input[type=checkbox]').forEach(function (cb) {{
      if (!cb.closest('.mark')) cb.checked = v;
    }});
    applyState();
  }}
  document.getElementById('ovall').addEventListener('click', function () {{ setAll(true); }});
  document.getElementById('ovnone').addEventListener('click', function () {{ setAll(false); }});
  SEL.addEventListener('change', applyState);
  document.querySelectorAll('.ovbar input[type=checkbox]')
          .forEach(function (cb) {{ cb.addEventListener('change', applyState); }});
</script>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser(description="Interactive report for one patient's model CSVs.")
    ap.add_argument("patient", help="patient id, e.g. 0010")
    ap.add_argument("--methods", default=None, help="comma list (default: every method found)")
    ap.add_argument("--open", action="store_true", help="open in the browser when done")
    a = ap.parse_args()

    models = find_models(a.patient, [m.strip() for m in a.methods.split(",")] if a.methods else None)
    if not models:
        print(f"No modeling CSVs for patient {a.patient} under {RUNS}/*/modeling/.\n"
              f"Run:  python run_pipeline.py --patient {a.patient} --methods all")
        sys.exit(1)

    fig, base, meta, order = build(models)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{a.patient}_model.html"
    out.write_text(page(a.patient, fig, base, models, meta, order), encoding="utf-8")
    print(f"{len(models)} method(s): {', '.join(models)}")
    print(f"wrote {out.relative_to(ROOT)}  ({out.stat().st_size/1e6:.1f} MB, opens offline)")
    if a.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
