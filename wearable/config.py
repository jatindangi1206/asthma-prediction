"""Every tunable constant and derived path, in one place."""
from pathlib import Path

# ---- Segmentation (H8) ---------------------------------------------------------
# A gap >= this opens a new chunk and flushes filter state. From the observed
# distribution of inter-reading gaps. Read by the chunker, every smoother, ingest
# (where NaN fillers go) and detect (per-chunk reset).
GAP_THRESHOLD_MIN = 180.0
FILL_INTERVAL_MIN = 10      # ingest writes a NaN-HRV filler at this cadence across a gap
MIN_SEGMENT_ROWS = 10       # a shorter chunk is left unsmoothed

# ---- CPD cold start ------------------------------------------------------------
BURN_IN_N = 144             # valid samples (24h at 10-min) to ARM, counted across chunks
POP_N0 = 30                 # population-prior pseudo-count (shrinkage strength)
POP_SIGMA0_DEFAULT = 15.0   # fallback prior if the cohort median is unavailable

# ---- Experiment defaults -------------------------------------------------------
# Held constant across a comparison, so only the method varies. The split comes from
# the sorted patient list + SEED alone, never from feature content.
SEED = 42
TEST_FRACTION = 0.3

# ---- Paths ---------------------------------------------------------------------
# data/ is partitioned by lifecycle, so a directory name says what stage it holds:
#   raw/        inputs, never written by the pipeline
#   processed/  ingest output
#   runs/       the artifact store — current results, each with its provenance
#   reports/    human-facing derived output (regenerable)
#   logs/       per-step run logs
#   old/        everything from previous runs — never read by the pipeline
DATA = Path("./data")
RAW_DIR = DATA / "raw"
PROCESSED_DIR = DATA / "processed"
RUNS_DIR = DATA / "runs"
REPORTS_DIR = DATA / "reports"
LOGS_DIR = DATA / "logs"
OLD_DIR = DATA / "old"
LABELS_CSV = PROCESSED_DIR / "labels_pass1.csv"

# Artifact table format: a setting, not a property of any step. Override per run with
# WEARABLE_FORMAT=parquet. A new backend is one entry in store._WRITERS/_READERS.
STORE_FORMAT = "csv"


# The pre-artifact-store per-patient layout, now under old/. The pipeline never writes
# here; only legacy/ and the comparison report's pf row read it.
def smoothed_dir(method: str) -> Path:
    return OLD_DIR / ("pf_smoothed" if method == "pf" else f"smoothed_{method}")


def calibrated_dir(method: str) -> Path:
    return OLD_DIR / f"calibrated_{method}"
