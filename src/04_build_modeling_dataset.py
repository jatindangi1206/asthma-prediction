"""Stage 04 — Build the ML-ready modeling dataset for ONE patient.

Consolidates a patient's entire physiological timeline into a single
timestamp-aligned CSV: raw channels (HR, sleep, SpO2, steps, temp, raw HRV) +
the chosen smoother's baseline/residual + the CPD annotations computed on that
residual + calendar features.

Run from the asthma-prediction/ directory:
    python src/04_build_modeling_dataset.py

================================================================================
MERGING PHILOSOPHY (why it is built this way)
================================================================================
1. MASTER TIMELINE = the 10-minute HRV grid.
   The annotated file (`data/annotated_<method>/<pid>_<method>_annotated.csv`)
   already contains the full grid — raw `hrvValue`, `smoothed_hrv`,
   `residual_hrv`, `chunk_id`, `gap_flag`, and every CPD column — produced on the
   exact residual the CPD ran on. We take it as the spine and align the OTHER
   raw channels onto its timestamps. This guarantees the smoother, the residual,
   and the annotations are never re-aligned (they were computed together).

2. CAUSAL ALIGNMENT (no future leakage).
   This feeds a real-time early-warning model, so a feature at time t may use
   only data with timestamp <= t. Every channel is therefore aligned with a
   BACKWARD-looking rule:
     • high-frequency channels (HR, temperature) -> aggregate the PRECEDING
       `window_min` window (mean/min/max) ending at t. Denoises and is causal.
     • interval-count channels (steps, distance)  -> SUM over the preceding
       window ending at t.
     • sparse spot channels (SpO2)                -> last value AS-OF t (backward
       merge) within a long tolerance, plus its staleness in minutes.
     • sleep sessions                              -> `is_asleep` = t falls inside
       a recorded [start, end] session; the 5-min epoch stage code as-of t.

3. KEEP THE FULL GRID (including gap-filler rows).
   Even where HRV is missing (gap_flag=1), HR/steps/sleep may exist and give the
   model context; the gap_flag lets the model mask. Nothing is interpolated
   across the 180-min voids — those are real `gap_flag=1` rows.

4. CALENDAR FEATURES.  time-of-day is sin/cos encoded (24 h period) so the model
   sees the circadian phase the residual was normalized against.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION  — set the run here
# ==============================================================================
PATIENT_ID        = "0010"        # patient to build
SMOOTHING_METHOD  = "gammadglm"   # which Track-A smoother's baseline/residual to use
ANNOTATION_METHOD = SMOOTHING_METHOD   # which annotated_<m> dir to pull CPD from
                                       #   (CPD ran on THIS method's residual)
CPD_FEATURE_MODE  = "all"         # "ensemble" | "all" | list e.g. ["bocpd","hmm"]

RAW_ROOT   = Path("raw_data")
SMOOTH_DIR = Path(f"data/smoothed_{SMOOTHING_METHOD}")
ANNOT_DIR  = Path(f"data/annotated_{ANNOTATION_METHOD}")
OUT_DIR    = Path("data/modeling")

# Channel registry — add a raw channel by adding one entry. Each is aligned onto
# the master HRV grid with the causal rule named in `align`.
RAW_CHANNELS = {
    "heart_rate": dict(subdir="heartrate",   time="logDateTime", value="lastRate",
                       align="window", window_min=10,
                       aggs={"hr_mean": "mean", "hr_min": "min", "hr_max": "max"}),
    "temperature": dict(subdir="temperature", time="createdTime", value="temperature",
                        align="window", window_min=10, aggs={"temp_mean": "mean"}),
    "steps":       dict(subdir="steps",       time="logDateTime", value="steps",
                        align="window", window_min=10, aggs={"steps_sum": "sum"}),
    "distance":    dict(subdir="steps",       time="logDateTime", value="distance",
                        align="window", window_min=10, aggs={"distance_sum": "sum"}),
    "spo2":        dict(subdir="spo2",        time="createdTime", value="spo2Value",
                        align="asof", tolerance_min=360, out="spo2"),
    # "sleep" is handled specially (interval membership) below.
}
SLEEP_SUBDIR = "sleep"
SLEEP_EPOCH_MIN = 5               # each tilde-separated code spans 5 minutes


# ==============================================================================
# RAW LOADERS
# ==============================================================================
def _patient_dir(pid):
    """Raw folders are named with quotes (e.g. \"0010\") or bare; resolve either."""
    for cand in (RAW_ROOT / f'"{pid}"', RAW_ROOT / pid):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No raw folder for patient {pid} under {RAW_ROOT}")


def _load_channel(pid, subdir, time_col, value_cols):
    """Concatenate all Spark part-CSVs in a channel subfolder; parse + sort."""
    d = _patient_dir(pid) / subdir
    parts = sorted(d.glob("part-*.csv"))
    if not parts:
        return None
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df.columns = df.columns.str.strip()
    cols = [time_col] + [c for c in np.atleast_1d(value_cols) if c in df.columns]
    df = df[[c for c in cols if c in df.columns]].copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    for c in np.atleast_1d(value_cols):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)


# ==============================================================================
# ALIGNERS  (all strictly causal: window ends at, or asof is backward from, t)
# ==============================================================================
def align_window(grid_ns, ch_time, ch_val, window_min, aggs):
    """Aggregate channel samples in the PRECEDING window (t-window, t] per grid t."""
    ct = ch_time.values.astype("datetime64[ns]")
    cv = ch_val.values.astype(float)
    win = np.timedelta64(int(window_min * 60), "s")
    lo = np.searchsorted(ct, grid_ns - win, side="right")
    hi = np.searchsorted(ct, grid_ns, side="right")
    out = {name: np.full(len(grid_ns), np.nan) for name in aggs}
    fns = {"mean": np.mean, "min": np.min, "max": np.max, "sum": np.sum,
           "median": np.median, "std": np.std, "count": len}
    for i in range(len(grid_ns)):
        a, b = lo[i], hi[i]
        if b > a:
            seg = cv[a:b]
            seg = seg[np.isfinite(seg)]
            if seg.size:
                for name, how in aggs.items():
                    out[name][i] = fns[how](seg)
    return out


def align_asof(master, ch, time_col, value_col, out_col, tolerance_min):
    """Last known value as-of t (backward) within tolerance; plus its age (min)."""
    left = master[["createdTime"]].sort_values("createdTime")
    right = ch[[time_col, value_col]].rename(columns={time_col: "createdTime"}).sort_values("createdTime")
    merged = pd.merge_asof(left, right, on="createdTime", direction="backward",
                           tolerance=pd.Timedelta(minutes=tolerance_min))
    merged = merged.rename(columns={value_col: out_col})
    # staleness: minutes since the matched reading
    age = pd.merge_asof(left, right.assign(_t=right["createdTime"]),
                        on="createdTime", direction="backward",
                        tolerance=pd.Timedelta(minutes=tolerance_min))["_t"]
    merged[f"{out_col}_age_min"] = (merged["createdTime"] - age).dt.total_seconds() / 60.0
    return merged.set_index("createdTime")[[out_col, f"{out_col}_age_min"]]


def align_sleep(pid, grid_ns):
    """is_asleep (inside a session interval) + the 5-min epoch stage code as-of t."""
    d = _patient_dir(pid) / SLEEP_SUBDIR
    parts = sorted(d.glob("part-*.csv"))
    is_asleep = np.zeros(len(grid_ns), int)
    stage = np.full(len(grid_ns), np.nan)
    if not parts:
        return is_asleep, stage
    sl = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    sl.columns = sl.columns.str.strip()
    sl["logDateTime"] = pd.to_datetime(sl["logDateTime"], errors="coerce")
    sl["logEndTime"] = pd.to_datetime(sl["logEndTime"], errors="coerce")
    sl = sl.dropna(subset=["logDateTime", "logEndTime"])
    epoch = np.timedelta64(int(SLEEP_EPOCH_MIN * 60), "s")
    for _, r in sl.iterrows():
        s = np.datetime64(r["logDateTime"]); e = np.datetime64(r["logEndTime"])
        lo = np.searchsorted(grid_ns, s, side="left")
        hi = np.searchsorted(grid_ns, e, side="right")
        if hi > lo:
            is_asleep[lo:hi] = 1
            # decode the per-epoch stage code active at each grid point in-session
            codes = str(r.get("description", "")).split("~")
            codes = [c for c in codes if c.strip() != ""]
            if codes:
                offs = ((grid_ns[lo:hi] - s) / epoch).astype(int)
                offs = np.clip(offs, 0, len(codes) - 1)
                stage[lo:hi] = [pd.to_numeric(codes[o], errors="coerce") for o in offs]
    return is_asleep, stage


# ==============================================================================
# CPD COLUMN SELECTION
# ==============================================================================
def cpd_columns(df, mode):
    ens = [c for c in ["anomaly_present", "primary_phenotype", "magnitude"] if c in df.columns]
    if mode == "ensemble":
        return ens
    detectors = ["bocpd", "kcpd", "kalman", "hmm"] if mode == "all" else list(mode)
    per = [f"{m}_{suf}" for m in detectors for suf in ("anomaly", "phenotype", "magnitude")
           if f"{m}_{suf}" in df.columns]
    return per + ens


# ==============================================================================
# BUILD
# ==============================================================================
def build_patient_dataset(pid, smoothing_method, annotation_method, cpd_mode):
    # 1) master spine = the annotated file (grid + smoothed + residual + CPD)
    annot = ANNOT_DIR / f"{pid}_{smoothing_method}_annotated.csv"
    if not annot.exists():
        raise FileNotFoundError(f"Annotated file not found: {annot} "
                                f"(run the smoother + 03_annotate {annotation_method} first)")
    master = pd.read_csv(annot)
    master.columns = master.columns.str.strip()
    master["createdTime"] = pd.to_datetime(master["createdTime"])
    master = master.sort_values("createdTime").reset_index(drop=True)
    grid_ns = master["createdTime"].values.astype("datetime64[ns]")

    out = pd.DataFrame({"timestamp": master["createdTime"], "patient_id": pid})

    # 2) HRV trio from the spine
    out["raw_hrv"] = master["hrvValue"]
    out["smoothed_hrv"] = master["smoothed_hrv"]
    out["residual_hrv"] = master["residual_hrv"]
    out["hrv_observed"] = master["hrvValue"].notna().astype(int)

    # 3) raw channels (causal alignment per the registry)
    for name, cfg in RAW_CHANNELS.items():
        ch = _load_channel(pid, cfg["subdir"], cfg["time"], cfg["value"])
        if ch is None or ch.empty:
            for col in (cfg.get("aggs", {}) or {cfg.get("out", name): None}):
                out[col] = np.nan
            continue
        if cfg["align"] == "window":
            agg = align_window(grid_ns, ch[cfg["time"]], ch[cfg["value"]],
                               cfg["window_min"], cfg["aggs"])
            for col, vals in agg.items():
                out[col] = vals
        elif cfg["align"] == "asof":
            a = align_asof(master, ch, cfg["time"], cfg["value"], cfg["out"], cfg["tolerance_min"])
            out = out.merge(a, left_on="timestamp", right_index=True, how="left")

    # 4) sleep (special: interval membership + epoch stage)
    is_asleep, sleep_stage = align_sleep(pid, grid_ns)
    out["is_asleep"] = is_asleep
    out["sleep_stage_code"] = sleep_stage

    # 5) calendar features (circadian phase)
    minute_of_day = out["timestamp"].dt.hour * 60 + out["timestamp"].dt.minute
    out["tod_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    out["tod_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    out["day_of_week"] = out["timestamp"].dt.dayofweek

    # 6) segmentation + CPD annotations from the spine
    for c in ("gap_flag", "chunk_id"):
        if c in master.columns:
            out[c] = master[c]
    for c in cpd_columns(master, cpd_mode):
        out[c] = master[c]

    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building modeling dataset — patient {PATIENT_ID}  smoother={SMOOTHING_METHOD}  "
          f"CPD={CPD_FEATURE_MODE}")
    df = build_patient_dataset(PATIENT_ID, SMOOTHING_METHOD, ANNOTATION_METHOD, CPD_FEATURE_MODE)
    out_path = OUT_DIR / f"{PATIENT_ID}__sm-{SMOOTHING_METHOD}__model.csv"
    df.to_csv(out_path, index=False)

    obs = df["hrv_observed"].sum()
    print(f"\nRows: {len(df)}  ({obs} HRV-observed, {len(df) - obs} gap-filler)")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"\nNon-null counts:\n{df.notna().sum().to_string()}")
    print(f"\nHead (observed rows):")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df[df.hrv_observed == 1].head(4).to_string(index=False))
    print(f"\nWrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()
