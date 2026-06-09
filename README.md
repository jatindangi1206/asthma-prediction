# Asthma Prediction from Wearable HRV

## Project Overview

This is a research pipeline for predicting asthma exacerbations from wearable
Heart-Rate-Variability (HRV) data. Raw recordings come from Samsung wearable
sensors (~110 patients); each patient's HRV signal is smoothed with a sequential
Monte-Carlo particle filter and then scanned by four change-point detection
(CPD) methods to locate physiological state transitions that may precede an
asthma episode. The accompanying paper is `main.tex`.

## Quick Start

Clone the repository and install the Python dependencies (developed on Python
3.10):

```bash
git clone <REPO_URL> asthma-prediction   # TODO: confirm from repo
cd asthma-prediction
pip install -r requirements.txt
```

Run the pipeline from the `asthma-prediction/` directory — the location of
`./data/` and `./raw_data/`. Each stage reads the previous stage's output:

```bash
python src/00_preprocess_raw.py
python src/01_hyper_tuning.py
python src/02_run_filters.py
python src/03_annotate.py
python scripts/visualize.py     # optional diagnostics
```

The data flows `raw_data/ -> data/processed/ -> data/smoothed/ -> data/annotated/`.

## Pipeline Stages

All scripts run from `asthma-prediction/`. The five stages are sequential; each
consumes the previous stage's CSV output.

1. **Preprocess** (`src/00_preprocess_raw.py`). Parses the raw Spark-partition
   HRV CSVs for the patient IDs listed in `raw_data/processed_users.txt`, sorts
   by time, and inserts NaN placeholder rows at 10-minute intervals wherever the
   gap between two real observations exceeds 180 minutes. Writes
   `data/processed/<pid>_processed.csv` with `createdTime`, `hrvValue`, and
   `minute_diff` columns. Key constants: `GAP_THRESHOLD_MIN=180`,
   `FILL_INTERVAL_MIN=10`.

2. **Hyperparameter tuning** (`src/01_hyper_tuning.py`). Sweeps the particle
   count N and the number of backward-sampling paths M per patient, using
   effective-sample-size (ESS) convergence, and recommends a single global N*
   and M* for the cohort. Writes `data/results/hyperparameter_log.csv`.

3. **SMC smoothing** (`src/02_run_filters.py`). Applies a Bootstrap particle
   filter with Forward-Filtering-Backward-Sampling (FFBS) to each patient. Set
   the `GLOBAL_N` and `GLOBAL_M` constants at the top of the file from stage 2
   before running. Writes `data/smoothed/<pid>_smoothed.csv` with `smoothed_hrv`
   (level + circadian), `true_trend_level` (level only, circadian removed), and a
   `gap_flag` column marking the NaN-inserted rows.

4. **CPD annotation** (`src/03_annotate.py`). Runs the four change-point
   detectors on `true_trend_level` (falling back to `smoothed_hrv` for older
   files). Writes `data/annotated/<pid>_annotated.csv` with a `<method>_type` and
   `<method>_degree` column per method, plus a `cpd_input_col` record of which
   column was scanned. This stage runs single-threaded.

5. **Visualization** (`scripts/visualize.py`, optional). Renders per-patient
   diagnostic PNGs — raw HRV, gap markers, smoothed signal with credible band,
   and the underlying trend with CPD shift markers — into `data/plots/`.

### State-space model

`HRVParticleModel` (defined identically in the stage-2 tuning and stage-3
smoothing scripts) is a 6-state SSM: a local linear trend (`level`, `slope`)
plus two harmonic seasonal components (`c1`/`c1_star` circadian, `c2`/`c2_star`
sub-circadian). The observable is `level + c1 + c2` with a Student-t observation
likelihood (robust to outliers); NaN observations use a near-flat likelihood so
the filter propagates on dynamics alone. Model parameters are data-driven per
patient (robust noise scale from the MAD of first differences, initial level
from the first-day median, frequencies from the median sampling interval).

### CPD methods

| Module | Algorithm | Output labels |
|---|---|---|
| `src/bocpd.py` | Bayesian Online CPD (Adams & MacKay 2007), O(T) Normal-Normal conjugate | normal / transition / shift |
| `src/kcp.py` | Kernel CPD via `ruptures` (RBF kernel) | normal / transition / shift |
| `src/kalman_cpd.py` | Kalman filter innovation score (nu^2 / S) | normal / medium / adverse |
| `src/hmm_cpd.py` | 2-state Gaussian HMM via `hmmlearn` | normal / transition / shift |

Each module exposes a single `detect(time, values) -> (change_types,
change_degrees)` interface. Stages 1, 2, and visualization parallelise across
`cpu_count - 1` workers.

## Repo Layout

```
asthma-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── CLAUDE.md
├── main.tex
├── src/
│   ├── 00_preprocess_raw.py
│   ├── 01_hyper_tuning.py
│   ├── 02_run_filters.py
│   ├── 03_annotate.py
│   ├── bocpd.py
│   ├── kalman_cpd.py
│   ├── kcp.py
│   ├── hmm_cpd.py
│   ├── eda_smoothed.py
│   └── patch_add_trend_level.py
├── scripts/
│   └── visualize.py
├── notebooks/
│   ├── 01_eda.ipynb
│   └── outputs/
└── tests/
    ├── __init__.py
    └── test_smoke.py
```

The `data/` and `raw_data/` directories live at the repo root but are largely
git-ignored (see `.gitignore`); only patient `0010` and the EDA summary files
are tracked as a worked example.

## Data Notes

- Raw patient folders have quoted names like `"0010"` — the preprocessing script
  handles both quoted and bare folder names.
- `raw_data/processed_users.txt` controls which patients are processed; entries
  are quoted IDs like `"0010"`.
- The `minute_diff` column computed in preprocessing carries through all stages
  as the irregular time step for the particle filter's `dt_norm`.

## Citation

TODO: confirm from repo. <!-- Add BibTeX / paper citation once main.tex is published. -->

## License

TBD
