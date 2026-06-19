# Asthma Exacerbation Early-Warning System — HRV Pipeline

> Continuous ambulatory **Heart-Rate-Variability** (HRV, 10-minute SDNN) from
> consumer smartwatches → a personalized, causal **early-warning signal** for
> asthma exacerbations.

Python · 10-min SDNN · ~109 patients · causal/online · fully reproducible

This repository turns a noisy, gap-ridden HRV stream into (a) a smooth
physiological **baseline** (Track A), (b) a mean-zero **residual** that exposes
autonomic volatility (Track B − A), and (c) **rich change-point annotations** —
the supervised target for a downstream ML model. Eleven smoothing methods and a
four-detector change-point ensemble are implemented behind one uniform interface
so any component can be swapped, re-tuned, or extended via a one-line registry
edit.

This README is the **engineering + research manual**: architecture, how every
stage runs, the complete parameter reference, **how and where to change every
constraint** (the 180-min chunking, the 24-h burn-in, every detector threshold,
every smoother knob), reproducibility, and a contribution/PR guide.

---

## Table of contents

1. [Architecture: the Dual-Track Residual pipeline](#1-architecture-the-dual-track-residual-pipeline)
2. [Repository layout](#2-repository-layout)
3. [Quickstart](#3-quickstart)
4. [Pipeline stages in detail](#4-pipeline-stages-in-detail)
5. [The 11 smoothing methods (Track A)](#5-the-11-smoothing-methods-track-a)
6. [The CPD ensemble (Track B) — detectors, phenotypes, fusion](#6-the-cpd-ensemble-track-b--detectors-phenotypes-fusion)
7. [Cold-start: cumulative burn-in + population prior](#7-cold-start-cumulative-burn-in--population-prior)
8. [⚙️ Every constraint and how to change it](#8-️-every-constraint-and-how-to-change-it)
9. [Output schemas & data dictionary](#9-output-schemas--data-dictionary)
10. [Evaluation: the smoother comparison report](#10-evaluation-the-smoother-comparison-report)
11. [Reproducibility](#11-reproducibility)
12. [Performance & scaling](#12-performance--scaling)
13. [Contributing & PR guide](#13-contributing--pr-guide)
14. [Roadmap & known limitations](#14-roadmap--known-limitations)
15. [Glossary, references, license](#15-glossary-references-license)

---

## 1. Architecture: the Dual-Track Residual pipeline

Running change-point detection (CPD) on **raw** HRV fails (circadian rhythm +
baseline drift cause false alarms); running CPD on **smoothed** HRV also fails
(it erases the volatility bursts that signal an attack). We resolve this with the
**Dual-Track Residual Architecture**:

```
 raw_data/<pid>/hrv/*.csv
        │  00_preprocess_raw.py   (parse Spark CSVs; insert NaN fillers; 180-min rule)
        ▼
 data/processed/<pid>_processed.csv
        │
        │            ┌──────────────── TRACK A (Context) ────────────────┐
        ├──►  02x_run_<method>.py  ──►  smoothed_hrv   (expected baseline) │
        │            │                                                    │
        │            │                    residual_hrv = hrvValue − smoothed_hrv
        │            │                                                    │
        └──────────── TRACK B (Reality) = raw hrvValue ───────────────────┘
        │
        ▼  data/smoothed_<method>/<pid>_<method>.csv     (carries residual_hrv)
        │
        ▼  03_annotate.py <method>     (CPD ensemble runs ON residual_hrv,
        │                               per-chunk reset, 24-h burn-in + pop prior)
        ▼
 data/annotated_<method>/<pid>_<method>_annotated.csv   ← Rich Annotation Vector = ML target Y
        │
        ▼  (downstream, out of scope here) multi-task Transformer/SSM:
           X = [smoothed_hrv, residual_hrv, steps, sleep, sin/cos(time_of_day)]
           Y = {anomaly_present, primary_phenotype, magnitude}
```

All methods obey a shared **H-Framework**: **H1** positivity, **H2/H4** drifting
multimodal baseline, **H5** circadian rhythm, **H7** irregular sampling, **H8**
≥180-min segmentation (state flush), **H9** strict causality (no peeking ahead),
**H10** per-patient adaptation. A plain-language guide to each method is in
`HRV_Smoothing_Methods_Explained.docx`.

---

## 2. Repository layout

```
asthma-prediction/
├── README.md                         ← this manual
├── CLAUDE.md                         ← state-space model spec / engineering notes
├── HRV_Smoothing_Methods_Explained.docx
├── requirements.txt                  ← pinned deps (Python 3.10)
├── main.tex                          ← the paper
├── raw_data/
│   ├── "0010"/ hrv/ … heartrate/ sleep/ spo2/ steps/ temperature/
│   └── processed_users.txt           ← quoted patient IDs to process
├── src/
│   ├── 00_preprocess_raw.py          ← Stage 0
│   ├── 01_hyper_tuning.py            ← Stage 1 (optional, particle-filter N*,M*)
│   ├── 02_run_filters.py             ┐
│   ├── 02b…02k_run_*.py              ┘ Stage 2 orchestrators (11 smoothers, runnable)
│   ├── <method>_smoother.py          ← 10 smoother libraries (smooth_dataframe + Config)
│   ├── 03_annotate.py                ← Stage 3: CPD ensemble orchestrator
│   ├── bocpd.py / kcp.py / kalman_cpd.py / hmm_cpd.py   ← 4 CPD detectors (detect())
│   └── eda.py / eda_smoothed.py      ← diagnostics
├── scripts/
│   ├── smoother_comparison_report.py     ← 11-method evaluation report (registry-driven)
│   ├── plot_annotated_interactive.py     ← per-patient residual + CPD interactive graph
│   └── plot_<method>.py / visualize*.py
└── data/
    ├── processed/          <pid>_processed.csv          (Stage 0)
    ├── smoothed/           <pid>_smoothed.csv           (Stage 2 — particle filter)
    ├── smoothed_<method>/  <pid>_<method>.csv           (Stage 2b–2k, Track A + residual)
    ├── annotated_<method>/ <pid>_<method>_annotated.csv (Stage 3, ML target Y)
    ├── plots/              comparison + per-patient interactive HTML
    └── results/            *_log.csv, hyperparameter_log.csv, eda/
```

---

## 3. Quickstart

All commands run **from the `asthma-prediction/` directory**.

```bash
# 0) environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Stage 0 — preprocess (needs raw_data/processed_users.txt)
python src/00_preprocess_raw.py

# 2) Stage 2 — Track A: run a smoother (pick one; run several to benchmark)
python src/02b_run_rspf.py        # RS-PF       -> data/smoothed_rspf/
python src/02g_run_gammadglm.py   # Gamma DGLM  -> data/smoothed_gammadglm/
python src/02k_run_ctimm.py       # CT-IMM      -> data/smoothed_ctimm/
#   (full list in §5)

# 3) Stage 3 — Track B: CPD ensemble on residual_hrv for a chosen Track A
python src/03_annotate.py rspf    # -> data/annotated_rspf/
#   method keys: pf rspf krlst gpssm ossa kim gammadglm logmhw hrkf anrewma  (ctimm via registry)

# 4) Evaluate smoothers side-by-side on one patient (standalone HTML report)
python scripts/smoother_comparison_report.py 0010   # -> data/plots/smoother_comparison_0010.html

# 5) Per-patient residual + CPD interactive graph
python scripts/plot_annotated_interactive.py rspf 0010
```

**Stop a run:** `Ctrl-C` (orchestrators use a process pool; partial per-patient
outputs already written stay on disk). Background runs: `pkill -f 03_annotate.py`.
Everything is **idempotent** — re-running overwrites per-patient outputs.

---

## 4. Pipeline stages in detail

| Stage | Script | Reads | Writes | Key constants |
|---|---|---|---|---|
| 0 Preprocess | `00_preprocess_raw.py` | `raw_data/`, `processed_users.txt` | `data/processed/` | `GAP_THRESHOLD_MIN=180`, `FILL_INTERVAL_MIN=10` |
| 1 Tune (opt.) | `01_hyper_tuning.py` | `data/processed/` | `results/hyperparameter_log.csv` | particle `N*`, `M*` |
| 2 Smooth | `02*_run_*.py` (×11) | `data/processed/` | `data/smoothed_<m>/` | per-method `GLOBAL_*` (§5) |
| 3 Annotate | `03_annotate.py <m>` | `data/smoothed_<m>/` | `data/annotated_<m>/` | `BURN_IN_N=144`, `POP_N0=30`, detector consts (§6) |
| Report | `smoother_comparison_report.py` | `data/processed/` (+`data/smoothed/` for PF) | `data/plots/` | registry (§10) |

**Stage 0** parses the per-patient Spark-partition HRV CSVs, sorts by time, and
inserts NaN-HRV filler rows every `FILL_INTERVAL_MIN` minutes across gaps larger
than `GAP_THRESHOLD_MIN`, so downstream stages can *see* the voids without
interpolating across them. Output columns: `createdTime, hrvValue, minute_diff`.

**Stage 2** (any of the 11 smoothers): keeps observed rows only, assigns
`chunk_id` at every ≥180-min gap, filters each chunk independently with state
reset (H8/H9), merges `smoothed_hrv` back onto the full grid, and computes
`residual_hrv = hrvValue − smoothed_hrv`.

**Stage 3** runs the four CPD detectors on `residual_hrv`, per chunk, after a
24-h burn-in, and writes the Rich Annotation Vector (§6).

---

## 5. The 11 smoothing methods (Track A)

All ten causal smoothers share one library contract:

```python
smooth_dataframe(df, patient_col, timestamp_col, value_col, config, out_col) -> DataFrame
#   df has [patient_id, timestamp, hrv_value]; returns it + chunk_id + <out_col> + residual_hrv
```

Each method has a `@dataclass` Config; the orchestrator exposes the smoothness/
robustness knobs as `GLOBAL_*` at the top of `02*_run_*.py`. **Structural knobs
were tuned on patient 0010's densest 3-day window**; **per-patient quantities**
(observed range, MAD-noise, log-mean, median dt) are auto-derived (H10).

| # | Method | Orchestrator → output | Tunable `GLOBAL_*` (defaults) |
|---|---|---|---|
| 1 | Particle filter (SMC+FFBS, **non-causal**) | `02_run_filters.py` → `smoothed/` | `N=250`, `M=50` |
| 2 | RS-PF (regime-switching PF) | `02b_run_rspf.py` → `smoothed_rspf/` | `K=3`, `N_PARTICLES=500` |
| 3 | KRLS-T (kernel RLS tracker) | `02c_run_krlst.py` → `smoothed_krlst/` | `BUDGET=100`, `LAMBDA=0.995` |
| 4 | GP-SSM (state-space GP) | `02d_run_gpssm.py` → `smoothed_gpssm/` | `Q_SCALE=0.05`, `R_SCALE=5.0` |
| 5 | OSSA (online SSA) | `02e_run_ossa.py` → `smoothed_ossa/` | `WINDOW_MAX=144`, `N_COMPONENTS=2` |
| 6 | Kim filter (Markov-switching SSM) | `02f_run_kim.py` → `smoothed_kim/` | `Q_SCALE=0.1`, `R_SCALE=3.0` |
| 7 | Gamma CD-DGLM | `02g_run_gammadglm.py` → `smoothed_gammadglm/` | `DELTREND=0.99`, `DELSEAS=0.99`, `SHAPE_SCALE=1.0` |
| 8 | Log-MHW (log multiplicative Holt-Winters) | `02h_run_logmhw.py` → `smoothed_logmhw/` | `ALPHA=0.08`, `GAMMA=0.02`, `OPTIMIZE=True` |
| 9 | H-RKF (Huber robust Kalman) | `02i_run_hrkf.py` → `smoothed_hrkf/` | `DELTREND=0.99`, `DELSEAS=0.99`, `HUBER_DELTA=1.345` |
| 10 | AN-REWMA (adaptive norm + robust EWMA) | `02j_run_anrewma.py` → `smoothed_anrewma/` | `NORM_ALPHA=0.05`, `EWMA_C=1.0` |
| 11 | CT-IMM (continuous-time interacting multiple model) | `02k_run_ctimm.py` → `smoothed_ctimm/` | `Q_SCALE=0.001`, `STICKINESS=0.95` |

> The original SMC particle filter (#1) uses FFBS **backward** smoothing → its
> `smoothed_hrv` and therefore its `residual_hrv` are **non-causal**; use it for
> offline comparison only. The ten causal methods (2–11) are the production set.

Each smoother CSV carries both `smoothed_hrv` and `residual_hrv`. A common
self-test runs from the module itself (e.g. `python src/ct_imm_smoother.py`) and
asserts the H1/H8/H9 guarantees on synthetic data.

---

## 6. The CPD ensemble (Track B) — detectors, phenotypes, fusion

The four detectors run **only on `residual_hrv`**, per `chunk_id` (state flushed
at gaps), after arming. Each exposes the contract:

```python
detect(time, values, priors) -> {"anomaly": 0/1[], "phenotype": str[], "magnitude": float[], "severity": float[]}
#   priors = {"m0": ..., "sigma0": ...}  (the per-patient burn-in noise floor, §7)
```

**Seven phenotypes** are emitted (the "Rich Annotation Vector"):

| Detector (file) | Phenotype | Fires when… | `magnitude` (native units) |
|---|---|---|---|
| **BOCPD** (`bocpd.py`) | `BOCPD_Mean_Shift` | run-length collapse, mean change dominates | actual Δmean (residual ms) |
| | `BOCPD_Variance_Change` | run-length collapse, spread change dominates | actual Δspread (residual ms) |
| **KCPD** (`kcp.py`) | `KCPD_Distribution_Shift` | studentized two-window MMD ≥ `MIN_EFFECT_Z` null-σ | MMD² distance |
| **Kalman** (`kalman_cpd.py`) | `Kalman_Volatility_Spike` | \|standardized innovation z\| > `CUSUM_H` | peak \|z\| |
| | `Kalman_Gradual_Drift` | two-sided CUSUM of innovations > `CUSUM_H` | CUSUM value |
| | `Kalman_Autocorrelation_Change` | rolling lag-1 innovation autocorr > `ACF_H` | \|lag-1 autocorr\| |
| **HMM** (`hmm_cpd.py`) | `HMM_State_Transition` | Viterbi decodes the high-variance State 1 | P(State 1) |

*(`KCP_Multivariate_Shift` is reserved for the future multi-channel extension —
a joint MMD over HRV+SpO₂+steps — and is inactive on a single residual channel.)*

**Detector constants** (defaults; all at the top of each file):

| File | Constants |
|---|---|
| `bocpd.py` | `HAZARD_LAMBDA=1000` (1/1000 hazard), `MU0=0.0`, `MIN_RUN=6`, `DROP_FRAC=0.5` |
| `kcp.py` | `WINDOW=18` (3 h/side), `N_PERM=80`, `STRIDE=3`, `MIN_EFFECT_Z=10.0`, `STUDENTIZE=True`, `STUD_HALFLIFE=36` |
| `kalman_cpd.py` | `R_SCALE=1.0`, `Q_FACTOR=1e-3`, `CUSUM_K=0.5`, `CUSUM_H=3.0`, `ACF_WIN=12`, `ACF_H=0.6` |
| `hmm_cpd.py` | `STICKINESS=0.95`, `STATE1_VAR_MULT=9.0`, `_MIN_ROWS=12` |

**Ensemble fusion (in `03_annotate.py`):** `anomaly_present` = **logical OR** of
the four detectors (maximally sensitive — recall-first for screening);
`primary_phenotype`/`magnitude` are taken from the firing method with the
**largest normalized severity**. Per-method columns
`{bocpd,kcpd,kalman,hmm}_{anomaly,phenotype,magnitude}` are also written. During
burn-in everything is `burn_in`/0. The HITL loop later down-weights any detector
that over-fires for a given patient (the per-method columns make this possible).

> Compare detectors by **distinct events** (contiguous runs), not flagged rows —
> the HMM marks every sample of a sustained regime, so its row count ≫ its event
> count. The annotation log reports both.

---

## 7. Cold-start: cumulative burn-in + population prior

Because residual noise floors vary by person and hardware (H10), and because
wearable compliance is **Missing-Not-At-Random** (devices come off during sleep/
charging), the calibration does **not** demand wall-clock continuity. Instead
(`03_annotate.py`):

- **Cumulative counting across chunks.** Valid residual samples are counted
  across *all* chunks, ignoring MNAR gaps. The first `BURN_IN_N = 144` valid
  samples (= 24 h of *active* observation, the minimum to span one diurnal cycle
  without overfitting a single phase) form the burn-in.
- **Bayesian shrinkage to a population prior.** The patient's own burn-in noise
  estimate `σ_obs` is shrunk toward a cohort-derived population prior `σ_pop`
  (median per-patient robust residual std) with pseudo-count `POP_N0 = 30`:
  `σ0² = (POP_N0·σ_pop² + n·σ_obs²)/(POP_N0 + n)`; the mean is shrunk toward 0.
  This keeps the noise floor robust when few early samples exist (the prior
  dominates) and converges to the patient's own as data accrues (at n=144,
  ~83 % patient weight). Fallback `POP_SIGMA0_DEFAULT = 15.0` ms.
- **Arming gate.** During accumulation the ensemble is **disarmed** (all alarms
  suppressed, phenotype `burn_in`); it arms only after the 144th valid sample.
  Patients with <144 valid samples never arm — they produce **no annotation**
  rather than a population-mean false alarm (verified on the 10-sample edge case).
- **σ0 injection.** The calibrated `σ0` seeds BOCPD's variance prior, the Kalman
  `R`, and the HMM emission variances.

The architecture is designed for **longitudinal evolution**: after this static
burn-in, the smoother itself (e.g. RS-PF) continuously updates the baseline
posterior, so the effective memory naturally grows beyond the 7–14-day clinical
norm toward a 90-day "healthspan" baseline.

---

## 8. ⚙️ Every constraint and how to change it

This is the canonical map of every knob, **where it lives**, and **how to change
it**. Constants are defined at the top of each file unless noted.

### 8.1 The 180-minute chunking (H8 segmentation)

The chunker `assign_chunks()` splits a patient series wherever the gap between
consecutive readings ≥ a threshold, and resets every model's state at the
boundary. The threshold appears in **three places** that must stay consistent:

| Where | Constant | Purpose |
|---|---|---|
| `src/00_preprocess_raw.py` | `GAP_THRESHOLD_MIN = 180` | where NaN fillers are inserted |
| every `src/<method>_smoother.py` | `GAP_THRESHOLD_MIN = 180.0` (+ `gap_threshold_min` in each Config) | per-chunk smoothing reset |
| `src/03_annotate.py` | `GAP_THRESHOLD_MIN = 180.0` | per-chunk CPD reset + `_ensure_chunk_id` |

- **Change the threshold** (e.g. to 120 or 240 min): edit all three. Easiest: set
  the value once and import it; or pass `gap_threshold_min=` into each Config.
- **Change the chunking *strategy*** (not just the threshold): replace the body
  of `assign_chunks()` (it is duplicated per smoother and in `03_annotate`). It
  currently emits `chunk_id = (gap ≥ threshold).cumsum()`. Alternatives you can
  drop in: **fixed-length windows** (`chunk_id = floor(elapsed_min / W)`),
  **sleep/wake segmentation** (boundary at sleep-stage transitions from the
  `sleep/` channel), or **activity-gated** (boundary when `steps` spikes). Keep
  the contract: a monotone integer `chunk_id` per row, resetting per patient.

### 8.2 The 24-hour burn-in / cold-start

All in `src/03_annotate.py`:

| Constant | Default | Change to… |
|---|---|---|
| `BURN_IN_N` | `144` (24 h) | `1008` (7 days) or `2016` (14 days) for the clinical multi-day baseline; smaller arms sooner but is noisier |
| `POP_N0` | `30` | larger ⇒ stronger shrinkage to the population prior (better for sparse patients) |
| `POP_SIGMA0_DEFAULT` | `15.0` ms | the fallback noise floor if no cohort is available |
| `_population_sigma0()` | median per-patient MAD-std | swap for a fixed clinical prior, or a per-device prior |

To make the burn-in **sleep-anchored** (start counting only after the first valid
sleep block), gate the `obs_idx` selection on the `sleep` channel before taking
the first `BURN_IN_N`.

### 8.3 Smoother knobs (Track A "stiffness")

Edit the `GLOBAL_*` at the top of the relevant `src/02*_run_*.py` (table in §5),
or pass overrides to the method's `Config`. General rule: **smaller process noise
/ larger obs noise / larger discount ⇒ smoother** (less residual volatility);
the opposite ⇒ more reactive. The HITL loop tunes these per patient.
`MIN_SEGMENT_ROWS = 10` (each smoother) is the shortest chunk that gets smoothed.

### 8.4 Detector thresholds (Track B sensitivity)

Edit the constants in §6 at the top of each detector file:

- **More/less sensitive overall:** BOCPD `HAZARD_LAMBDA`, `DROP_FRAC`; KCPD
  `MIN_EFFECT_Z` (raise ⇒ fewer alarms); Kalman `CUSUM_H`/`ACF_H`; HMM
  `STICKINESS` (raise ⇒ fewer regime flips).
- **KCPD speed vs. resolution:** `STRIDE` (raise to speed up large patients),
  `N_PERM` (lower to speed up); `STUDENTIZE`/`STUD_HALFLIFE` control the
  circadian-variance removal.
- **Ensemble vote:** change the fusion rule in `03_annotate.py` (currently
  `anomaly_present = OR` of the four). For a majority rule, require ≥2 firing; for
  weighted HITL, scale each `{m}_severity` before the argmax.

### 8.5 Preprocessing

`src/00_preprocess_raw.py`: `FILL_INTERVAL_MIN = 10` (filler cadence — match your
sampling rate), `GAP_THRESHOLD_MIN` (see §8.1). Which patients are processed is
controlled by `raw_data/processed_users.txt`.

---

## 9. Output schemas & data dictionary

**Smoothed** — `data/smoothed_<method>/<pid>_<method>.csv`:

```
createdTime, hrvValue, minute_diff, smoothed_hrv, residual_hrv, gap_flag, chunk_id
```

| Column | Meaning |
|---|---|
| `hrvValue` | raw 10-min SDNN (Track B); NaN on gap-filler rows |
| `smoothed_hrv` | Track A baseline (>0) |
| `residual_hrv` | `hrvValue − smoothed_hrv` (NaN where raw missing) |
| `gap_flag` | 1 on inserted filler rows, else 0 |
| `chunk_id` | segment id, +1 at each ≥180-min gap |

**Annotated** — `data/annotated_<method>/<pid>_<method>_annotated.csv`: the
smoothed columns **plus**

```
{bocpd,kcpd,kalman,hmm}_anomaly,  {…}_phenotype,  {…}_magnitude,
anomaly_present, primary_phenotype, magnitude, cpd_input_col(=residual_hrv)
```

**Logs** (`data/results/`): `smoothing_<method>_log.csv` (per-patient status,
`n_chunks`, `min_smoothed`), `annotation_<method>_log.csv` (per-patient
`n_armed`, `n_anomaly`, per-method event counts, burn-in `sigma0`),
`phase1_preprocess_log.csv`, `hyperparameter_log.csv`.

---

## 10. Evaluation: the smoother comparison report

`scripts/smoother_comparison_report.py <pid>` runs **every registered smoother**
on one patient and emits a **single standalone interactive HTML** (Plotly
embedded — works offline): a summary table (runtime, variance↓, roughness↓,
mean-error, peak-preservation, correlation, chunks, obs→smoothed) + metric
definitions + a synchronized stack of raw-vs-smoothed charts.

**Modular registry** — add a new smoother to the report by appending one entry to
`REGISTRY` (module name, Config class, params string). A `source="precomputed"`
entry can read an existing smoothed CSV (used for the slow non-causal PF). Metric
definitions are documented in-report; "roughness reduction" is stated neutrally
(no value judgment — selection is the reader's call).

---

## 11. Reproducibility

- **Environment:** Python **3.10**; `pip install -r requirements.txt`. Key deps:
  `numpy, pandas, scipy, scikit-learn` (GP kernels, HMM scaling), `statsmodels`
  (Holt-Winters), `hmmlearn` (HMM CPD), `particles` (SMC), `matplotlib`+`plotly`,
  `tqdm`. `ruptures` is legacy (the current `kcp.py` is self-contained).
- **Determinism:** every stochastic component is seeded — particle filters use
  fixed RNG seeds in their configs; `kcp.py` uses `np.random.default_rng(0)` for
  the MMD permutations; HMM uses `random_state=42`. Given identical inputs and
  constants, outputs are byte-reproducible. The causal smoothers additionally
  pass a truncation test (a prefix run is identical to the full run on shared
  rows — the H9 guarantee, asserted in each module's `__main__`).
- **Reproduce a result:** Stage 0 → the chosen `02*` smoother → `03_annotate.py
  <method>`. Each stage is idempotent and overwrites per-patient files. The
  comparison report is one command (§10).
- **Pin your run:** record the smoother `GLOBAL_*`, the detector constants, and
  `BURN_IN_N/POP_N0/POP_SIGMA0_DEFAULT`; they fully determine the annotation.

---

## 12. Performance & scaling

- **Parallelism:** Stages 0/1 and all smoothers use a `ProcessPoolExecutor`
  (`cpu_count − 1` workers). Stage 3 runs patient-by-patient.
- **Cost hot-spots:** KCPD's permutation test is the heaviest CPD step
  (O(T · N_PERM · w²)); it is strided (`STRIDE`) and gated. For very large
  patients raise `STRIDE`/lower `N_PERM` (a small resolution trade-off). RS-PF and
  OSSA are the slowest smoothers (particles / per-step SVD); GP-SSM, Log-MHW and
  AN-REWMA are sub-second per patient.
- **Typical timings (10–20 k observed pts):** smoothing 0.2–20 s, annotation
  6–25 s per patient.

---

## 13. Contributing & PR guide

### Add a new smoothing method (Track A)
1. Create `src/<name>_smoother.py` exposing `smooth_dataframe(df, patient_col,
   timestamp_col, value_col, config, out_col)` and an `@dataclass <Name>Config`.
   It **must**: assign `chunk_id` (reuse `assign_chunks`), filter each chunk
   independently with a fresh state (H8), be causal (H9), output positive values
   (H1), derive per-patient params (H10), and add `residual_hrv`.
2. Add a `__main__` self-test asserting H1 (min>0), H8 (chunk count), H9
   (prefix==full truncation).
3. Create `src/02x_run_<name>.py` (copy a sibling orchestrator; set `GLOBAL_*`,
   output dir, file suffix; compute `residual_hrv`).
4. Register it in `scripts/smoother_comparison_report.py` `REGISTRY` (one line)
   and, if it feeds CPD, add it to `SMOOTHERS` in `src/03_annotate.py`.
5. Tune `GLOBAL_*` on patient 0010's densest window; document the defaults.

### Add a new CPD detector (Track B)
1. Create `src/<name>_cpd.py` exposing `detect(time, values, priors) ->
   {anomaly, phenotype, magnitude, severity}` (per-row arrays). Use
   `priors["sigma0"]`; be causal; return native-unit `magnitude` and a [0,1]
   `severity` for the ensemble argmax.
2. Add `("<name>", module)` to `METHODS` in `src/03_annotate.py`; the per-method
   columns and event counts wire up automatically.
3. Add its phenotype colors to `scripts/plot_annotated_interactive.py`.

### Conventions & review checklist
- Causality is non-negotiable for production methods (no future leakage). Any
  backward/offline pass (FFBS, RTS smoother) must be labeled non-causal.
- Keep the 180-min chunk reset and the per-patient calibration intact.
- Prefer data-driven per-patient priors over hard-coded scales.
- Update this README's parameter tables (§5/§6/§8) when adding constants.
- Run the module self-tests and the comparison report before opening a PR.

---

## 14. Roadmap & known limitations

- **Multivariate CPD:** `KCP_Multivariate_Shift` activates once SpO₂/steps/sleep
  are fused into a residual *vector* (joint MMD).
- **Online longitudinal baseline:** wire the RS-PF posterior into a rolling
  90-day "healthspan" baseline (replacing the static burn-in after calibration).
- **HITL Bayesian optimization:** auto-tune Track-A stiffness and per-method
  voting weights from patient-reported outcomes.
- **Self-supervised pretraining:** mask 4-h `residual_hrv` windows and reconstruct
  them before fine-tuning on the annotations.
- **Limitations today:** the SMC particle filter is non-causal; KCPD can retain a
  faint residual circadian tilt; patients with <144 valid samples are
  un-annotatable by design; the downstream ML model is out of scope here.

---

## 15. Glossary, references, license

**Glossary.** *Residual* = raw − smoothed (the autonomic deviation CPD analyzes).
*Track A/B* = context baseline / raw reality. *Causal/online* = uses only
past+present. *Chunk* = a continuous segment between ≥180-min gaps. *Burn-in* =
the first 144 valid samples used to calibrate noise priors. *Phenotype* = the
categorical signature of a detected change. *Magnitude* = native-unit severity.

**Key references.** Adams & MacKay (2007, BOCPD, arXiv:0710.3742); Gretton et al.
(2012, MMD, JMLR 13); Page (1954, CUSUM, Biometrika); Roberts (1959, EWMA);
Rabiner (1989, HMM, Proc. IEEE); Blom & Bar-Shalom (1988, IMM, IEEE TAC);
West & Harrison (1997, DGLM); Aminikhanghahi & Cook (2017, CPD survey). Per-method
sources are in each module docstring and in `HRV_Smoothing_Methods_Explained.docx`.

**License / data.** Research preview. HRV data is sensitive — do not commit
`raw_data/` or any patient CSVs to a public remote.
