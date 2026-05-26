"""Central configuration: paths, the cross-stage column contract, and constants.

Paths derive from this file's location (``Path(__file__).parents``) so the repo
can be cloned anywhere. A new patient is just a new folder under
``data/raw_data/`` — nothing here is patient-specific.

Stage data flow (each stage reads the previous stage's CSV):
    data/raw_data/<id>/        -> Stage 1 preprocessing -> <id>_aligned.csv
    data/processed/<id>_aligned.csv  -> Stage 2 cleaning  -> <id>_cleaned.csv
    data/processed/<id>_cleaned.csv  -> Stage 3 smoothing -> <id>_smoothed.csv
    data/processed/<id>_smoothed.csv -> Stage 4 cpd       -> outputs/<id>_final.csv
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PIPELINE_DIR = Path(__file__).resolve().parents[0]          # <repo>/pipeline
REPO_ROOT = Path(__file__).resolve().parents[1]             # <repo>

DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw_data"                        # patient folders live here
PROCESSED_DIR = DATA_DIR / "processed"                      # intermediate stage CSVs

CHECKPOINTS_DIR = PIPELINE_DIR / "checkpoints"
OUTPUTS_DIR = PIPELINE_DIR / "outputs"
VAE_CHECKPOINT = CHECKPOINTS_DIR / "vae_cnn.pt"


def processed_path(patient_id: str, stage: str) -> Path:
    """``data/processed/<patient_id>_<stage>.csv`` (stage in aligned|cleaned|smoothed)."""
    return PROCESSED_DIR / f"{_safe(patient_id)}_{stage}.csv"


def final_path(patient_id: str) -> Path:
    """``pipeline/outputs/<patient_id>_final.csv``."""
    return OUTPUTS_DIR / f"{_safe(patient_id)}_final.csv"


def _safe(patient_id: str) -> str:
    """Filesystem-safe patient id (folder names may contain quotes/spaces)."""
    import re
    return re.sub(r"[^A-Za-z0-9_-]", "", patient_id) or "patient"


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Column contract shared across stages
# --------------------------------------------------------------------------- #
# Stage 1 emits these intermediate names (kept through stages 2-3); Stage 4
# renames the working columns to the final contract on output.
MINUTE_COL = "minute"            # integer-minute axis from common start
TIMESTAMP_COL = "timestamp"      # real datetime, carried for traceability
HRV_RAW_COL = "hrvValue"         # raw HRV -> becomes real_value in the final CSV
HRV_CLEAN_COL = "hrv_clean"      # Stage 2 output: mean-filled + gap-chunked HRV
SMOOTHED_COL = "smoothed_value"  # Stage 3 output: VAE reconstruction of HRV
MINUTE_DIFF_COL = "minute_diff"  # Stage 2 working column
MEAN_GAP_COL = "mean"            # Stage 2 working column (file-level mean gap)

# Final-CSV renames (working name -> contract name).
FINAL_RENAME = {MINUTE_COL: "timestep", HRV_RAW_COL: "real_value"}

# Working/scratch columns that must NOT appear in the final CSV (everything else
# on the smoothed frame is treated as context and carried through automatically).
INTERNAL_COLS = (HRV_CLEAN_COL, MINUTE_DIFF_COL, MEAN_GAP_COL)

# Context signal columns (carried untouched; never smoothed or detected on).
CONTEXT_SIGNALS = ("HR", "steps", "temperature", "spo2", "is_asleep")

# --------------------------------------------------------------------------- #
# Stage 1 — preprocessing constants
# --------------------------------------------------------------------------- #
WATCH_OFF_TEMP_THRESHOLD = 90.0    # temp (degF) below this => sensor off-skin
# clip_flag marks raw HRV outside a plausible physiological range (flag only;
# real_value is never altered). Tune as needed.
HRV_CLIP_MIN = 0.0
HRV_CLIP_MAX = 300.0

PART_GLOB = "part-*.csv"
IGNORE_NAMES = ("_SUCCESS",)
IGNORE_SUFFIXES = (".crc",)

# --------------------------------------------------------------------------- #
# Stage 2 — cleaning constants
# --------------------------------------------------------------------------- #
# Gaps in the minute axis larger than the file mean gap split the HRV series
# into chunks; straddling values are masked to NaN and never imputed.
GAP_MULTIPLIER = 1.0               # threshold = GAP_MULTIPLIER * mean(minute_diff)

# --------------------------------------------------------------------------- #
# Stage 3 — smoothing / VAE hyperparameters (defaults — tune later)
# --------------------------------------------------------------------------- #
# Architecture matches the pasted CNN-VAE: two Conv1d layers (channels_1,
# channels_2), kernel 5, full-resolution latent.
#
# Window/stride tuned to MEASURED segment-length distribution after chunking.
# Patient 10: max segment = 108, so window 128 has 0% coverage; window 16 gives
# 78% coverage (see `python -m pipeline.characterize <id>` to re-measure).
VAE_WINDOW = 16                # was 128 — fits the median-9 segment regime
VAE_STRIDE = 4                 # was 16 — proportional to the smaller window
VAE_LATENT = 8
VAE_CHANNELS = (32, 64)        # (channels_1, channels_2)
VAE_KERNEL = 5
VAE_EPOCHS = 40
VAE_BATCH_SIZE = 64
VAE_LR = 1e-3
VAE_BETA = 0.01
VAE_SEED = 42

# --------------------------------------------------------------------------- #
# Stage 4 — CPD constants (indicators 0/1, magnitudes in [0, 1])
# --------------------------------------------------------------------------- #
BOCPD_HAZARD = 1 / 250.0
BOCPD_PROB_THRESHOLD = 0.3

KCP_PENALTY = 10.0
KCP_MIN_SIZE = 10

HMM_N_STATES = 3
HMM_SEED = 42

KALMAN_PROCESS_VAR = 1e-3
KALMAN_MEAS_VAR = 1.0
KALMAN_RESID_THRESHOLD = 3.0
