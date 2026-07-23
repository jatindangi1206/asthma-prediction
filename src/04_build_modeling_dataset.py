"""Stage 04 — Build the ML-ready modeling dataset for ONE patient.

Consolidates a patient's physiological timeline into a single timestamp-aligned
CSV: the raw channels (heart rate, temperature, steps, SpO2, sleep stage, raw
HRV) + the chosen smoother's baseline/residual + the CPD annotations computed on
that residual.

We do NOT invent or aggregate anything. Each channel keeps its ORIGINAL column
name and its ORIGINAL recorded value; we only place that value on the master
timeline at the right time.

Run from the asthma-prediction/ directory:
    python src/04_build_modeling_dataset.py

================================================================================
MERGING LOGIC
================================================================================
1. MASTER TIMELINE = the 10-minute HRV grid.
   The annotated file (`data/annotated_<method>/<pid>_<method>_annotated.csv`)
   already holds the full grid — raw `hrvValue`, `smoothed_hrv`, `residual_hrv`,
   `gap_flag`, `chunk_id`, and every CPD column — produced on the exact residual
   the CPD ran on. It is the spine; the other raw channels are placed onto its
   `createdTime` timestamps. The smoother / residual / annotations are never
   re-aligned (they were computed together).

2. PLACING A CHANNEL'S VALUE (causal, no future leakage).
   This feeds a real-time early-warning model, so a value at time t may use only
   data with timestamp <= t. Each non-HRV channel is joined with a BACKWARD
   as-of merge: the most recent recorded reading at or before t, within
   ASOF_TOLERANCE_MIN (so a reading is not carried across a long gap). The
   ORIGINAL value is written under the channel's ORIGINAL column name:
        heart rate  -> lastRate
        temperature -> temperature
        steps       -> steps
        SpO2        -> spo2Value

3. SLEEP STAGE.
   The sleep file stores, per session [logDateTime, logEndTime], a `description`
   string of 5-minute stage codes (e.g. "176~30~30~80~120~..."). For each grid
   timestamp inside a session we write the actual code active at that 5-minute
   epoch into `sleep_stage`. Timestamps outside any session are left blank (no
   sleep was recorded — nothing is invented).

4. KEEP THE FULL GRID. Rows where HRV is missing keep `gap_flag=1`; nothing is
   interpolated across the 180-min voids.
"""

from __future__ import annotations

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
CPD_FEATURE_MODE  = "all"         # "ensemble" | "all" | list e.g. ["bocpd","hmm"]

ASOF_TOLERANCE_MIN = 60           # max minutes a reading may be carried forward to a grid point

RAW_ROOT   = Path("raw_data")
ANNOT_DIR  = Path(f"data/annotated_{ANNOTATION_METHOD}")
OUT_DIR    = Path("data/modeling")

# Channel registry — each raw channel is placed on the grid by a backward as-of
# merge. The ORIGINAL value column name is kept as the output column name.
RAW_CHANNELS = {
    "heartrate":   dict(subdir="heartrate",   time="logDateTime",  value="lastRate"),
    "temperature": dict(subdir="temperature", time="createdTime",  value="temperature"),
    "steps":       dict(subdir="steps",       time="logDateTime",  value="steps"),
    "spo2":        dict(subdir="spo2",         time="createdTime",  value="spo2Value"),
}
SLEEP_SUBDIR = "sleep"
SLEEP_EPOCH_MIN = 5               # each tilde-separated code spans 5 minutes (verified in data)


# ==============================================================================
# RAW LOADERS
# ==============================================================================
def _patient_dir(pid):
    """Raw folders are named with quotes (e.g. \"0010\") or bare; resolve either."""
    for cand in (RAW_ROOT / f'"{pid}"', RAW_ROOT / pid):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No raw folder for patient {pid} under {RAW_ROOT}")


def _load_channel(pid, subdir, time_col, value_col):
    """Concatenate all Spark part-CSVs in a channel subfolder; parse + sort by time."""
    d = _patient_dir(pid) / subdir
    parts = sorted(d.glob("part-*.csv"))
    if not parts:
        return None
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df.columns = df.columns.str.strip()
    if time_col not in df.columns or value_col not in df.columns:
        return None
    df = df[[time_col, value_col]].copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    return df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)


# ==============================================================================
# ALIGNERS  (strictly causal: backward as-of from t)
# ==============================================================================
def align_asof(grid, ch, time_col, value_col):
    """Most recent recorded value at or before each grid timestamp, within tolerance.
    Returns a Series aligned to `grid` (no aggregation; the original value is kept)."""
    left = pd.DataFrame({"createdTime": grid}).sort_values("createdTime")
    right = (ch[[time_col, value_col]]
             .rename(columns={time_col: "createdTime"})
             .sort_values("createdTime"))
    merged = pd.merge_asof(left, right, on="createdTime", direction="backward",
                           tolerance=pd.Timedelta(minutes=ASOF_TOLERANCE_MIN))
    return merged.set_index("createdTime")[value_col]


def align_sleep(pid, grid):
    """The 5-minute sleep-stage code active at each grid timestamp (blank if no
    session covers it). Writes the ACTUAL recorded code, nothing decoded."""
    d = _patient_dir(pid) / SLEEP_SUBDIR
    parts = sorted(d.glob("part-*.csv"))
    grid_ns = np.asarray(grid, dtype="datetime64[ns]")
    stage = np.full(len(grid_ns), np.nan)
    if not parts:
        return stage
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
            codes = [c for c in str(r.get("description", "")).split("~") if c.strip() != ""]
            if codes:
                offs = np.clip(((grid_ns[lo:hi] - s) / epoch).astype(int), 0, len(codes) - 1)
                stage[lo:hi] = [pd.to_numeric(codes[o], errors="coerce") for o in offs]
    return stage


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
    # 1) master spine = the annotated file (grid + raw/smoothed/residual HRV + CPD)
    annot = ANNOT_DIR / f"{pid}_{smoothing_method}_annotated.csv"
    if not annot.exists():
        raise FileNotFoundError(f"Annotated file not found: {annot} "
                                f"(run the smoother + 03_annotate {annotation_method} first)")
    master = pd.read_csv(annot)
    master.columns = master.columns.str.strip()
    master["createdTime"] = pd.to_datetime(master["createdTime"])
    master = master.sort_values("createdTime").reset_index(drop=True)
    grid = master["createdTime"]

    # 2) spine columns, original names
    out = pd.DataFrame({"createdTime": grid, "patient_id": pid})
    out["hrvValue"] = master["hrvValue"]
    out["smoothed_hrv"] = master["smoothed_hrv"]
    out["residual_hrv"] = master["residual_hrv"]

    # 3) raw channels — actual value placed at each timestamp (backward as-of)
    for name, cfg in RAW_CHANNELS.items():
        ch = _load_channel(pid, cfg["subdir"], cfg["time"], cfg["value"])
        if ch is None or ch.empty:
            out[cfg["value"]] = np.nan
        else:
            out[cfg["value"]] = align_asof(grid, ch, cfg["time"], cfg["value"]).values

    # 4) sleep stage code (actual 5-min code active at t)
    out["sleep_stage"] = align_sleep(pid, grid)

    # 5) segmentation + CPD annotations from the spine
    #    candidate_event = Stage-03a post-gap "keep-but-verify" flag (present only if 03a ran);
    #    carried unconditionally like gap_flag/chunk_id — it's a calibration flag, not a detector.
    for c in ("gap_flag", "chunk_id", "candidate_event"):
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

    obs = int(df["hrvValue"].notna().sum())
    print(f"\nRows: {len(df)}  ({obs} HRV-observed, {len(df) - obs} gap-filler)")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"\nNon-null counts:\n{df.notna().sum().to_string()}")
    print(f"\nHead (HRV-observed rows):")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(df[df.hrvValue.notna()].head(4).to_string(index=False))
    print(f"\nWrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()
