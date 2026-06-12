# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research pipeline for predicting asthma exacerbations using wearable HRV (Heart Rate Variability) data. Raw data comes from Samsung wearable sensors (~110 patients), each with folders containing heartrate, hrv, sleep, spo2, steps, and temperature CSVs in Spark partition format. The pipeline processes HRV signals through smoothing and change-point detection to identify physiological state transitions that may precede asthma episodes.

The accompanying paper is `main.tex`.

## Pipeline Stages

All scripts are run from the `asthma-prediction/` directory (where `./data/` and `./raw_data/` live). Each stage reads from the previous stage's output:

```
raw_data/  →  data/processed/  →  data/smoothed/  →  data/annotated/
```

1. **Preprocess** – parse raw Spark CSVs, fill gaps >180 min with NaN placeholders at 10-min intervals:
   ```
   python src/00_preprocess_raw.py
   ```
   Requires `raw_data/processed_users.txt` listing patient IDs. Outputs `data/processed/<pid>_processed.csv`.

2. **Hyperparameter tuning** – find optimal particle filter N (particles) and M (backward-sampling paths) per cohort using ESS convergence:
   ```
   python src/01_hyper_tuning.py
   ```
   Reads from `data/` directory, outputs `data/results/hyperparameter_log.csv` with recommended global N* and M*.

3. **SMC smoothing** – apply Bootstrap particle filter with FFBS (Forward Filtering Backward Sampling) to smooth HRV, independently per contiguous observed segment. Update `GLOBAL_N` and `GLOBAL_M` constants in `02_run_filters.py` from step 2 results:
   ```
   python src/02_run_filters.py
   ```
   Outputs `data/smoothed/<pid>_smoothed.csv` with `smoothed_hrv`, `true_trend_level`, and `gap_flag` columns.

4. **CPD annotation** – run four change-point detection methods on smoothed signal:
   ```
   python src/03_annotate.py
   ```
   Outputs `data/annotated/<pid>_annotated.csv` with `{method}_type` and `{method}_degree` columns.

5. **Visualization** (optional):
   ```
   python scripts/visualize.py
   ```
   Generates 4-panel diagnostic PNGs in `data/plots/`.

6. **EDA** (optional, diagnostic) – plot-first, descriptive-only characterisation of the raw processed signal; grounds the H1–H10 assumptions the smoother relies on. No hypothesis tests.
   ```
   python src/eda.py            # whole cohort
   python src/eda.py 0010       # one patient
   ```
   Outputs to `data/results/eda/`: per-patient 6-panel figures, `cohort_eda.png`, `per_patient.csv`, and the `h1_h10.csv` verdict table.

## Architecture

### State-Space Model (`HRVParticleModel`)
Defined identically in both `01_hyper_tuning.py` and `02_run_filters.py`. A 5-state SSM in **logit space**, grounded in the cohort EDA (`src/eda.py`, H1–H10):
- `level`: local-level random walk (baseline drift). There is **no `slope`** term — H4 showed drift is mild, and a deterministic slope extrapolated smoothed values far outside the data range.
- `c1`, `c1_star`, `c2`, `c2_star`: two harmonic seasonal components (circadian + sub-circadian), kept because H5 circadian is strong.

Observable = `lo + (hi - lo) * sigmoid(level + c1 + c2)`, where `[lo, hi]` is the patient's exact observed `[min, max]`. The sigmoid confines every smoothed value strictly inside the data range and absorbs the ceiling saturation from H1 (HRV is bounded — cohort-wide [22, 129] — and piles up at the ceiling). Observation noise is **Gaussian** (H6: within-segment first differences are near-normal, so the previous Student-t was unjustified).

The filter runs **independently per contiguous observed segment** (segments split at gap-fill NaN rows); segments shorter than `MIN_SEGMENT_ROWS=10` are left NaN. Nothing is interpolated across a >180-min gap, and each segment contains no NaN, so a plain `ssm.Bootstrap` is used (no custom NaN likelihood).

Parameters are data-driven per segment via `compute_data_driven_params()`: latent spreads from the MAD of the logit-transformed signal, `sigma_obs` from the MAD of raw first differences, frequencies from the median sampling interval. Outputs `smoothed_hrv` = `sigmoid(level+c1+c2)` and `true_trend_level` = `sigmoid(level)` (circadian removed), both in HRV units inside `(lo, hi)`.

### CPD Modules (called by `03_annotate.py`)
Each module in `src/` exposes a single `detect(time, values) -> (change_types, change_degrees)` interface:

| Module | Algorithm | Output labels |
|---|---|---|
| `bo_cpd.py` | Bayesian Online CPD (Adams & MacKay 2007), O(T) rolling Normal-Normal conjugate | normal / transition / shift |
| `k_cpd.py` | Kernel CPD via `ruptures` (RBF kernel) | normal / transition / shift |
| `kalman_cpd.py` | Kalman filter innovation score (nu²/S) | normal / medium / adverse |
| `hmm_cpd.py` | 2-state Gaussian HMM via `hmmlearn` | normal / transition / shift |

### Parallelism
Stages 1, 2, and visualization use `concurrent.futures.ProcessPoolExecutor` with `cpu_count - 1` workers. Stage 3 (annotation) runs single-threaded.

## Data Notes

- Raw patient folders have quoted names like `"0010"` — the preprocessing script handles both quoted and bare folder names.
- `processed_users.txt` controls which patients are processed; entries are quoted IDs like `"0010"`.
- The `minute_diff` column computed in preprocessing carries through all stages as the irregular time step for the particle filter's `dt_norm`.

## Key Constants to Tune

- `00_preprocess_raw.py`: `GAP_THRESHOLD_MIN=180`, `FILL_INTERVAL_MIN=10`
- `02_run_filters.py`: `GLOBAL_N` and `GLOBAL_M` (set from tuning results)
- `bo_cpd.py`: `_HAZARD_LAMBDA`, `_CP_THRESHOLD_TRANSITION`, `_CP_THRESHOLD_SHIFT`
- `kalman_cpd.py`: `_THRESHOLD` (innovation score cutoff)
- `k_cpd.py`: `_PENALTY` (lower = more change points)

## Dependencies

Key packages: `particles` (SMC/particle filters), `ruptures` (kernel CPD), `hmmlearn`, `numpy`, `pandas`, `matplotlib`, `tqdm`, `scikit-learn`.
