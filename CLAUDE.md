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

3. **SMC smoothing** – apply Bootstrap particle filter with FFBS (Forward Filtering Backward Sampling) to smooth HRV. Update `GLOBAL_N` and `GLOBAL_M` constants in `02_run_filters.py` from step 2 results:
   ```
   python src/02_run_filters.py
   ```
   Outputs `data/smoothed/<pid>_smoothed.csv` with `smoothed_hrv` column.

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

## Architecture

### State-Space Model (`HRVParticleModel`)
Defined identically in both `01_hyper_tuning.py` and `02_run_filters.py`. A 6-state SSM:
- `level` + `slope`: local linear trend
- `c1`, `c1_star`, `c2`, `c2_star`: two harmonic seasonal components (circadian + sub-circadian)

Observable = `level + c1 + c2`. Observation noise is Student-t (robust to outliers). NaN observations use a near-flat likelihood (`scale=1e6`) so the filter propagates on dynamics only.

Parameters are data-driven per patient via `compute_data_driven_params()`: robust noise scale from MAD of first differences, initial level from first-day median, frequencies from the median sampling interval.

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
