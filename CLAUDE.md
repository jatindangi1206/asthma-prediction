# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research pipeline that turns noisy, gap-ridden ambulatory **HRV** (10-min SDNN from
consumer smartwatches, ~109 patients) into a supervised **early-warning signal for
asthma exacerbations**. The downstream ML model is out of scope; this repo produces
its training target. The paper is `main.tex`; `README.md` is the full engineering +
research manual (parameter reference, contribution guide) and is the source of truth
when this file and it disagree.

**Run everything from the `asthma-prediction/` directory** — scripts use relative
`data/` and `raw_data/` paths. Not a git repo at the top level.

## Dual-Track Residual Architecture

The core idea: CPD on **raw** HRV false-alarms on circadian/drift; CPD on **smoothed**
HRV erases the volatility bursts that signal an attack. So:

- **Track A (context):** a smoother produces `smoothed_hrv`, the expected baseline.
- **Track B (reality):** raw `hrvValue`.
- **`residual_hrv = hrvValue − smoothed_hrv`** is what change-point detection runs on.

```
raw_data/<pid>/hrv/*.csv
  └─ 00_preprocess_raw.py        → data/processed/<pid>_processed.csv
       └─ 02x_run_<method>.py    → data/smoothed_<method>/<pid>_<method>.csv   (Track A + residual_hrv)
            └─ 03_annotate.py <method> → data/annotated_<method>/<pid>_<method>_annotated.csv  (ML target Y)
```

All methods obey a shared **H-Framework** (documented per-module and in
`HRV_Smoothing_Methods_Explained.docx`): H1 positivity, H2/H4 drifting multimodal
baseline, H5 circadian, H7 irregular sampling, **H8 ≥180-min segmentation (state
flush)**, **H9 strict causality (no peeking ahead)**, H10 per-patient adaptation.

## Commands

```bash
# environment (Python 3.10)
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Stage 0 — preprocess (needs raw_data/processed_users.txt)
python src/00_preprocess_raw.py

# Stage 1 — optional particle-filter hyperparameter tuning (N*, M*)
python src/01_hyper_tuning.py

# Stage 2 — run one smoother (one orchestrator per method, see table below)
python src/02g_run_gammadglm.py        # → data/smoothed_gammadglm/

# Stage 3 — CPD ensemble on residual_hrv for a chosen Track A method
python src/03_annotate.py gammadglm    # method key; default if omitted is "gammadglm"

# Smoother self-test (asserts H1/H8/H9 on synthetic data)
python src/ct_imm_smoother.py

# Evaluation: all smoothers on one patient → standalone interactive HTML
python scripts/smoother_comparison_report.py 0010

# Per-patient residual + CPD interactive graph
python scripts/plot_annotated_interactive.py rspf 0010

# EDA (whole cohort or one patient)
python src/eda.py            # or:  python src/eda.py 0010
```

Tests: `pytest` runs `tests/test_smoke.py` (import smoke test only — no real
coverage). Re-running any stage is idempotent (overwrites per-patient files). Stop a
background run with `pkill -f 03_annotate.py`.

## Smoothers (Track A) — `src/<method>_smoother.py` + `src/02x_run_<method>.py`

Every causal smoother library exposes one contract:

```python
smooth_dataframe(df, patient_col, timestamp_col, value_col, config, out_col) -> DataFrame
#   assigns chunk_id, filters each chunk independently with fresh state (H8),
#   causal (H9), positive output (H1), per-patient-derived params (H10), adds residual_hrv
```

Each method has an `@dataclass` Config; the runnable `02x_run_*.py` orchestrator
exposes the smoothness/robustness knobs as `GLOBAL_*` constants at the top. Structural
knobs were tuned on patient 0010's densest 3-day window; per-patient quantities
(observed range, MAD-noise, log-mean, median dt) are auto-derived.

| Key | Method | Orchestrator | Output dir |
|---|---|---|---|
| `pf` | Particle filter (SMC+FFBS, **non-causal — offline only**) | `02_run_filters.py` | `smoothed/` |
| `rspf` | Regime-switching PF | `02b_run_rspf.py` | `smoothed_rspf/` |
| `krlst` | Kernel RLS tracker | `02c_run_krlst.py` | `smoothed_krlst/` |
| `gpssm` | State-space GP | `02d_run_gpssm.py` | `smoothed_gpssm/` |
| `ossa` | Online SSA | `02e_run_ossa.py` | `smoothed_ossa/` |
| `kim` | Kim Markov-switching SSM | `02f_run_kim.py` | `smoothed_kim/` |
| `gammadglm` | Gamma CD-DGLM | `02g_run_gammadglm.py` | `smoothed_gammadglm/` |
| `logmhw` | Log multiplicative Holt-Winters | `02h_run_logmhw.py` | `smoothed_logmhw/` |
| `hrkf` | Huber robust Kalman | `02i_run_hrkf.py` | `smoothed_hrkf/` |
| `anrewma` | Adaptive-norm robust EWMA | `02j_run_anrewma.py` | `smoothed_anrewma/` |
| `ctimm` | Continuous-time IMM | `02k_run_ctimm.py` | `smoothed_ctimm/` |

The `pf` particle filter uses FFBS backward smoothing → its residual is **non-causal**;
use for offline comparison only. The other ten are the production set.

## CPD Ensemble (Track B) — `src/03_annotate.py`

Four detectors, registered in `METHODS = [("bocpd", bocpd), ("kcpd", kcp),
("kalman", kalman_cpd), ("hmm", hmm_cpd)]`, each exposing:

```python
detect(time, values, priors) -> {"anomaly", "phenotype", "magnitude", "severity"}  # per-row arrays
#   priors = {"m0", "sigma0"} from the per-patient burn-in noise floor
```

| File | Algorithm | Phenotypes |
|---|---|---|
| `bocpd.py` | Bayesian Online CPD (Adams & MacKay 2007) | `BOCPD_Mean_Shift`, `BOCPD_Variance_Change` |
| `kcp.py` | Studentized two-window MMD (self-contained; `ruptures` is legacy) | `KCPD_Distribution_Shift` |
| `kalman_cpd.py` | Kalman innovation CUSUM / autocorr | `Kalman_Volatility_Spike`, `_Gradual_Drift`, `_Autocorrelation_Change` |
| `hmm_cpd.py` | 2-state Gaussian HMM (`hmmlearn`) | `HMM_State_Transition` |

**Fusion:** `anomaly_present` = logical **OR** of the four (recall-first). The
`primary_phenotype`/`magnitude` come from the firing detector with the largest
normalized severity. Per-method `{bocpd,kcpd,kalman,hmm}_{anomaly,phenotype,magnitude}`
columns are written too (the HITL loop uses them to down-weight over-firing detectors).
**Compare detectors by distinct events (contiguous runs), not flagged rows** — the HMM
marks every sample of a sustained regime.

**Cold-start (in `03_annotate.py`):** valid residual samples are counted *cumulatively
across chunks* (ignoring MNAR gaps); the first `BURN_IN_N=144` (24h) calibrate the
noise floor `sigma0`, which is Bayesian-shrunk toward a cohort population prior with
pseudo-count `POP_N0=30` (fallback `POP_SIGMA0_DEFAULT=15.0` ms). The ensemble is
**disarmed** (`phenotype="burn_in"`) until armed at sample 144; patients with <144
valid samples never arm and produce **no annotation** (by design, not a bug).

## The 180-min chunk threshold lives in THREE places — keep consistent

`GAP_THRESHOLD_MIN` appears in `src/00_preprocess_raw.py` (where NaN fillers go), in
**every** `src/<method>_smoother.py` (per-chunk smoothing reset), and in
`src/03_annotate.py` (per-chunk CPD reset). Changing the segmentation means editing all
three. `assign_chunks()` emits `chunk_id = (gap ≥ threshold).cumsum()` and is duplicated
per smoother — preserve the contract: monotone integer `chunk_id` per row, reset per
patient.

## Key constants to tune

- `00_preprocess_raw.py`: `GAP_THRESHOLD_MIN=180`, `FILL_INTERVAL_MIN=10`
- each `02x_run_*.py`: per-method `GLOBAL_*` (see README §5); `MIN_SEGMENT_ROWS=10`
- `03_annotate.py`: `BURN_IN_N`, `POP_N0`, `POP_SIGMA0_DEFAULT`
- detectors: `bocpd.py` `HAZARD_LAMBDA`/`DROP_FRAC`; `kcp.py` `MIN_EFFECT_Z`/`STRIDE`/`N_PERM`; `kalman_cpd.py` `CUSUM_H`/`ACF_H`; `hmm_cpd.py` `STICKINESS`

## Conventions

- **Causality (H9) is non-negotiable** for production methods. Any backward/offline pass
  (FFBS, RTS) must be explicitly labeled non-causal. Smoother self-tests assert a
  truncation guarantee (prefix run == full run on shared rows).
- Prefer data-driven per-patient priors over hard-coded scales (H10).
- Determinism: particle filters seed RNG in their configs; `kcp.py` uses
  `np.random.default_rng(0)`; HMM uses `random_state=42`. Given identical inputs +
  constants, outputs are byte-reproducible.
- Adding a smoother/detector: see README §13 (also register in the comparison-report
  `REGISTRY` and in `SMOOTHERS`/`METHODS` of `03_annotate.py`).

## Data notes

- Raw patient folders have quoted names like `"0010"`; preprocessing handles both quoted
  and bare names. `raw_data/processed_users.txt` (quoted IDs) controls which patients run.
- **HRV data is sensitive — never commit `raw_data/` or any patient CSVs to a public remote.**
- Parallelism: Stages 0/1 and all smoothers use `ProcessPoolExecutor` (`cpu_count − 1`);
  Stage 3 runs patient-by-patient. KCPD's permutation test is the heaviest CPD step.
