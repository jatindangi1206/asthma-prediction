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

# ---- CPD settings, per channel -----------------------------------------------------
# Change-point detection runs on each channel separately, through a per-patient noise
# model (`cpd/_whiten.py`) that whitens the residual first. Everything about TIME is in
# minutes and converted per channel; nothing is counted in readings. `detect` holds the
# noise model's and the step's settings; the others are the detectors' (see each
# module's DEFAULTS). A channel or detector missing here falls back to those DEFAULTS.
# Every value is in the detect cache key, so editing one recomputes only that channel.
# Measured cadence: HR one reading a minute, HRV one every ~10 min (irregular). HR is
# median-binned to 5 min first (`bin_min`), so its re-entry gap is 3 bins (15 min).
CPD_SETTINGS = {
    "hrv": {
        "detect": {"step_min": 10.0, "burn_in_min": 1440.0, "baseline_days": 14.0,
                   "state_window_min": 10.0,
                   "abnormal_frac": 0.3, "reentry_gap_min": 60.0, "settle_min": 20.0,
                   "settle_h8_min": 30.0, "null_days": 60.0, "recal_days": 7.0,
                   "bin_min": 0.0},
        "bocpd":  {"lambda_min": 4320.0, "window_min": 60.0, "max_hyp": 300},
        "kcpd":   {"half_min": 180.0, "stride_min": 30.0, "coverage": 0.7,
                   "bin_min": 0.0, "n_perm": 80},
        "kalman": {"cusum_k": 0.5, "acf_window_min": 120.0, "coverage": 0.7},
        "hmm":    {"dwell_normal_min": 1440.0, "dwell_volatile_min": 120.0,
                   "state1_var_mult": 9.0},
        "drift":  {"window_start_h": 18, "window_end_h": 12, "baseline_nights": 28,
                   "min_nights": 3, "min_sleep_minutes": 60.0, "direction": -1,
                   "cusum_k": 0.5},
    },
    "hr": {
        "detect": {"step_min": 1.0, "burn_in_min": 1440.0, "baseline_days": 14.0,
                   "state_window_min": 1.0,
                   "abnormal_frac": 0.3, "reentry_gap_min": 15.0, "settle_min": 30.0,
                   "settle_h8_min": 30.0, "null_days": 60.0, "recal_days": 7.0,
                   "bin_min": 5.0},
        "bocpd":  {"lambda_min": 4320.0, "window_min": 60.0, "max_hyp": 300},
        "kcpd":   {"half_min": 180.0, "stride_min": 30.0, "coverage": 0.7,
                   "bin_min": 0.0, "n_perm": 80},
        "kalman": {"cusum_k": 0.5, "acf_window_min": 120.0, "coverage": 0.7},
        "hmm":    {"dwell_normal_min": 1440.0, "dwell_volatile_min": 120.0,
                   "state1_var_mult": 9.0},
        "drift":  {"window_start_h": 18, "window_end_h": 12, "baseline_nights": 28,
                   "min_nights": 3, "min_sleep_minutes": 60.0, "direction": 1,
                   "cusum_k": 0.5},
    },
}
# How often the whole detector ensemble may fire on no-change noise built from the
# patient's own data, per worn day (1440 worn minutes). Split evenly across detectors,
# and across a detector's statistics. No labels are involved. The experiment compares
# all three; CPD_BUDGET is the default.
CPD_BUDGETS = (1 / 7, 1 / 3, 1.0)
CPD_BUDGET = 1 / 3
# How the no-change data for those thresholds is built. "shuffle" (the default) resamples
# whole days of the patient's own past; "sts" simulates from a structural time-series
# model fitted on that past (`cpd/_sts.py`), on the same observation mask, through the
# same smoother and noise model. STS_SETTINGS are keyed only when "sts" is used, so the
# default's cache keys are untouched.
CPD_CALIBRATION = "shuffle"
CPD_CALIBRATIONS = ("shuffle", "sts")
STS_SETTINGS = {
    "hrv": {"grid_min": 10.0, "harmonics": 2, "fit_days": 7.0, "min_days": 3.0,
            "lo": 1.0, "hi": 129.0, "maxiter": 60},          # the device caps HRV at 129
    "hr":  {"grid_min": 1.0, "harmonics": 2, "fit_days": 7.0, "min_days": 3.0,
            "lo": 25.0, "hi": 230.0, "maxiter": 60},
}

# ---- Experiment defaults -------------------------------------------------------
# Held constant across a comparison, so only the method varies. The split comes from
# the sorted patient list + SEED alone, never from feature content.
SEED = 42
TEST_FRACTION = 0.2        # 80:20, the one split ratio used everywhere

# ---- Modelling experiments -----------------------------------------------------
# The experimental design: two tasks (detection, onset), a patient-level split drawn
# at several train:test ratios and several seeds, one fixed flag threshold, and
# patient-resampled intervals. A comparison varies the input (or the model) and holds
# all of this fixed.
THRESHOLD = 0.5             # a patient-day is flagged if its probability >= this
TEST_PCT = (20,)            # one ratio: 80:20. More ratios only widened the
                            # intervals — with ~10 test episodes a fold, a 50% test
                            # share leaves about 4, and the interval swamps the effect.
SEEDS = (0, 1, 2, 3, 4)     # one split per (TEST_PCT, seed) pair
BOOT = 1000                 # resamples of test patients for the 95% intervals (0 = none)
# The metrics an interval is reported for. The rest are point estimates: a bootstrapped
# median lead time is not meaningful when few episodes are caught, and resampling every
# metric makes a sweep several times slower for no extra reading.
BOOT_METRICS = ("auroc", "sensitivity", "specificity", "ppv", "brier")
FALSE_ALARM_DAYS = 30       # false alarms are reported per this many negative days
CALIBRATION_BANDS = 10      # probability bands of the calibration table

# ---- The two axes of the window ablation ---------------------------------------
# How much history the model sees, and how far ahead onset is asked about. Both are
# properties of the DATASET, so `join` owns them; a model only ever sees the columns it
# is handed.
#
# DAYS is the input window: the classified day plus the DAYS-1 days before it, as lag
# columns (`<feature>_d0` is the classified day, `_d1` yesterday, ...). DAYS = 1 is the
# classified day alone.
#
# HORIZON is how the ingested labels define onset: a new episode within 1-3 days. A
# smaller horizon asks a stricter question of the SAME days — `horizon=1` is "does an
# episode start tomorrow?" — and narrows the label without touching the sample set, so
# two horizons are read side by side (the reference design's rule).
DAYS = 1                    # default input window, in days
# Time-of-day slots per day of raw signal a modelling table carries, 0 = none. Day
# features are four summary statistics per day; a profile is the day itself, which is
# what a sequence or foundation model needs. 144 = the cohort's native 10-minute
# cadence, so DAYS=7 PROFILE=144 is a 1,008-point series per row. Off by default
# because those columns go to every model in the run.
PROFILE = 0
PROFILE_BINS = 144          # the setting to use when profiling: the native cadence
HORIZON = 3                 # onset: a new episode on d+1 ... d+HORIZON, as the labels hold it
WINDOW_DAYS = (7, 6, 5, 4, 3, 2, 1)   # the window ablation: 7 days of input down to one
WINDOW_HORIZON = (1,)       # onset ablation: the next day only (add 2, 3 to widen it)

# Columns a modelling table carries that are never features: the identifiers, the label,
# the episode id an event metric needs and the cohort group a slice is reported by.
# Defined here because join writes them, model must not train on them and evaluate
# scores with them — and no step may import another.
META_COLS = ("p_id", "date", "y", "event", "group")

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

# ---- Signal channels ------------------------------------------------------------
# Every physiological channel the pipeline can carry, and the column it is stored in.
# HRV keeps the legacy column name `hrvValue` so that every artifact, report and cache
# key built before the pipeline became multi-channel stays valid.
#
# Smoothing, CPD and the day features are all channel-agnostic: a smoother already
# takes `value_col` as an argument, so `smooth hr` runs the same ten methods over heart
# rate with no change to any of them. What differs per channel is its CADENCE and its
# GAPS, so each channel gets its own processed file built on its own observation times
# — a 180-minute hole in HR and a 180-minute hole in HRV are different events, and H8
# must segment each on its own.
CHANNELS = {
    # continuous physiological signals — a level worth estimating, so smoothable
    "hrv": "hrvValue", "hr": "hr", "temp": "temp", "spo2": "spo2", "steps": "steps",
    # sleep and activity. Ingested and turned into day features like anything else,
    # but NOT smoothable: `sleep_active` and `steps_active` are 0/1 state flags on
    # ~100% of raw rows, and the sleep stages are sparse (~0.3%) 5-minute durations
    # that include 0. A smoother must emit a strictly positive baseline (H1) and a
    # flag has no level to estimate, so `SMOOTHABLE` gates them out of smooth,
    # forecast and encode while `join` still gets "hours asleep", "minutes of deep
    # sleep" and sleep fragmentation out of them.
    "sleep_active": "sleep_active", "steps_active": "steps_active",
    "deep_sleep": "deep_sleep", "light_sleep": "light_sleep",
    "almost_awake": "almost_awake",
}
DEFAULT_CHANNEL = "hrv"
# Channels a smoother, forecaster or encoder will accept: continuous and positive.
SMOOTHABLE = ("hrv", "hr", "temp", "spo2", "steps")
# The channels an experiment PATH processes (smooth / forecast / detect / encode), each
# on its own and each its own feature block. `sweep ... --signals hrv` narrows it.
PATH_SIGNALS = ("hrv", "hr")
# What `join`'s raw baseline reads when nothing else is asked for.
# EVERY MODEL GETS EVERY VARIABLE. The question is never "can HRV alone predict this" —
# it is "can these measurements predict this". So join reads all of them by default and
# a path only decides what happens to the SMOOTHABLE ones; the rest always ride along
# raw. spo2 is left out because it has 21 readings in the whole cohort.
DEFAULT_JOIN_CHANNELS = ("hrv", "hr", "temp", "steps", "sleep_active",
                         "steps_active", "deep_sleep", "light_sleep", "almost_awake")

# How a channel becomes a day feature.
#   level     mean / SD / min / max — a reading is a measurement of a level
#   state     0/1 per minute: total minutes in the state, and how many separate bouts
#   duration  minutes already totalled per sleep bout by the device, so a day gets the
#             SUM and the bout count. Never averaged, and never spread across a
#             timeline — the export records "this bout had 45 min deep, 20 light" and
#             gives no start or end for any stage, so none is assumed.
STATE_CHANNELS = ("sleep_active", "steps_active")
DURATION_CHANNELS = ("deep_sleep", "light_sleep", "almost_awake")


def processed_path(pid: str, channel: str = DEFAULT_CHANNEL) -> Path:
    """Where one patient-channel's ingested series lives.

    HRV is `<pid>_processed.csv`, exactly where it has always been. Every other channel
    gets `<pid>__<channel>_processed.csv` — the double underscore is the repo's own
    field separator (as in `<pid>__<step>_<method>`), so a channel can never be mistaken
    for part of a patient id.
    """
    if channel not in CHANNELS:
        raise KeyError(f"unknown channel {channel!r}; available: {sorted(CHANNELS)}")
    stem = pid if channel == DEFAULT_CHANNEL else f"{pid}__{channel}"
    return PROCESSED_DIR / f"{stem}_processed.csv"


def processed_files(channel: str = DEFAULT_CHANNEL) -> list[Path]:
    """Every ingested file for one channel, sorted by patient id."""
    if channel not in CHANNELS:
        raise KeyError(f"unknown channel {channel!r}; available: {sorted(CHANNELS)}")
    out = []
    for f in sorted(PROCESSED_DIR.glob("*_processed.csv")):
        pid, _, chan = f.stem[: -len("_processed")].partition("__")
        if (chan or DEFAULT_CHANNEL) == channel:
            out.append(f)
    return out


def processed_pid(path: Path) -> str:
    """The patient id of a processed file, whichever channel it holds."""
    return Path(path).stem[: -len("_processed")].partition("__")[0]

# Artifact table format: a setting, not a property of any step. Override per run with
# WEARABLE_FORMAT=parquet. A new backend is one entry in store._WRITERS/_READERS.
STORE_FORMAT = "csv"


# The pre-artifact-store per-patient layout, now under old/. The pipeline never writes
# here; only legacy/ and the comparison report's pf row read it.
def smoothed_dir(method: str) -> Path:
    return OLD_DIR / ("pf_smoothed" if method == "pf" else f"smoothed_{method}")


def calibrated_dir(method: str) -> Path:
    return OLD_DIR / f"calibrated_{method}"
