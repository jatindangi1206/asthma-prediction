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

## 1. Architecture: five independent steps

```
sources ─▶ ingest ─▶ smooth ──┐
               │              ├─▶ join ─▶ model ─▶ evaluate
               └─▶ detect ────┘
                   (on raw, smoothed or residual)
```

No step imports another. `wearable/experiment.py` is the only module that connects
them, so each runs on its own and they compose in any order — or not at all: handing
the raw signal straight to the model is one call.

### The five rules the repo enforces

`python -m wearable selftest` and `tests/test_smoke.py` assert all of these.

1. **Steps do not depend on each other.** A test greps for cross-step imports.
2. **Adding a method is one file, nothing else.** Pools are discovered from the
   filesystem; there is no list to register in.
3. **Every result records how it was made** — method, settings, input artifacts and
   code version — in a `manifest.json` beside it, so any result traces back to the raw
   signal (`python -m wearable lineage ...`).
4. **The same computation is never run twice.** Artifacts are keyed on a content hash
   of exactly those things; a matching key is reused. Editing one method's source
   invalidates that method and nothing else.
5. **A comparison varies only the method.** Same model, same seed, and the same test
   participants — the split is derived from the patient universe and the seed alone,
   never from feature content.

### Signal choice: raw, smoothed, or residual

CPD on **raw** HRV false-alarms on circadian rhythm and baseline drift. CPD on
**smoothed** HRV erases the volatility bursts that signal an attack. The resolution is
the residual:

- a smoother produces `smoothed_hrv`, the expected baseline;
- `residual_hrv = hrvValue − smoothed_hrv`.

That rationale still stands, but the residual is now **one option of three**, not the
architecture. `detect --on raw|smoothed|residual` picks the column, and
`python -m wearable sweep signal` measures which actually wins. Some detectors work
better on a raw signal and some on a de-trended one, so this is a parameter.

---

## 2. Repository layout

Four pools, discovered from the filesystem. Drop a file in one and it is registered.

```
asthma-prediction/
├── README.md  CLAUDE.md  requirements.txt  main.tex
├── wearable/                    code: the package
├── scripts/                     code: runnable reporting tools
├── legacy/                      code: retired (see legacy/README.md)
├── tests/                       code: tests
├── docs/                        prose: method notes, the PDF and DOCX
└── data/                        partitioned by LIFECYCLE, never committed
    ├── raw/                     inputs, never written by the pipeline
    │   ├── pass1/               labelled 221-patient cohort (has its own README)
    │   └── spark/               older unlabelled export
    ├── processed/               ingest output + labels_pass1.csv
    ├── runs/<step>/<method>/<key>/   the artifact store — CURRENT results, each
    │                            with its manifest.json provenance record
    ├── reports/                 human-facing derived output (regenerable)
    │   ├── eda/
    │   ├── plots/
    │   └── smoother_comparison/
    ├── logs/                    per-step run logs (*_log.csv)
    └── old/                     everything from previous runs (see its README);
                                 never read by the pipeline
```

### Naming rule

Lowercase `snake_case` for every file and directory. Top-level directories are named by
**kind** (code / prose / data); everything under `data/` is named by **lifecycle stage**,
so a directory name tells you what stage produced it and whether it is current:
`runs/` is current and provenance-tracked, `reports/` is regenerable, `old/` holds
previous runs and is never read by the pipeline. Artifact files are `<pid>__<step>_<method>.<ext>` — no spaces, capitals,
parentheses or plus signs.

### How a method is registered

A module in a pool directory is a method if it exposes the pool's entry point. Optional
module attributes: `LABEL`, `DEFAULTS` (its knobs — these replaced the old `GLOBAL_*`
constants), `CONFIG` (its dataclass), `KIND` and `HIGHER_IS_BETTER` (metrics). A module
starting with `_`, or lacking the entry point, is skipped, so helpers can live
alongside. A method that fails to import is reported unavailable rather than taking the
registry down.

---

## 3. Quickstart

All commands run **from the `asthma-prediction/` directory**.

```bash
# 0) environment
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
#    Use `.venv/bin/python` directly: `source .venv/bin/activate` does not always take
#    in a non-interactive shell and falls back to a system python whose numpy is
#    x86_64 (architecture ImportError on Apple Silicon).

python -m wearable list       # every discovered method and its defaults
python -m wearable selftest   # assert the five rules; touches no data

# 1) ingest — any registered source, one output contract
python -m wearable ingest                    # default: the labelled pass1 cohort
python -m wearable ingest --source spark     # the legacy unlabelled cohort

# 2) run any step on its own
python -m wearable smooth gammadglm [--deltrend 0.995] [--force]
python -m wearable detect --on raw                       # CPD on the unsmoothed signal
python -m wearable detect --on residual --smoothing gammadglm --methods bocpd,kcpd

# 3) join picks whatever results you already have, and does the split
python -m wearable join --artifacts smooth:gammadglm,detect:bocpd --target detection
python -m wearable model join:smooth:gammadglm --model logreg
python -m wearable evaluate model:logreg

# 4) or chain them — any subset, including none
python -m wearable chain --smoothing gammadglm --cpd bocpd,kcpd --on residual
python -m wearable chain                     # no smoothing, no CPD: the raw baseline

# 5) compare methods, everything else held fixed
python -m wearable sweep smoothing           # every smoother + a no-smoothing baseline
python -m wearable sweep cpd                 # every detector, same signal and model
python -m wearable sweep signal              # one ensemble on raw vs smoothed vs residual
python -m wearable sweep smoothers-no-model  # judge smoothers with no model at all
#   add --limit 40 to cap patients for a fast sweep (it is part of the cache key)

# 6) provenance
python -m wearable runs                      # the artifact log
python -m wearable lineage model:logreg      # back to the raw signal
```

**Reuse:** re-running anything with the same method, settings, input and code version
prints `reusing <key>` and recomputes nothing. `--force` overrides.

**Storage format** is a setting, not code: `WEARABLE_FORMAT=parquet python -m wearable ...`
(or `config.STORE_FORMAT`). Adding a backend is one entry in `store._WRITERS`/`_READERS`.

**Stop a run:** `Ctrl-C`; completed artifacts stay. An interrupted artifact is left
`status: running` and is never reused, so a resumed run recomputes only it.

---

## 4. The steps in detail

| Step | Input | Output table(s) | Notes |
|---|---|---|---|
| `ingest` | a registered source | `data/processed/<pid>_processed.csv`, `labels_pass1.csv` | the only step that still writes per-patient files; parallel |
| `smooth` | ingest | `smoothed` | adds `residual_hrv`; H8 chunking, H1 positivity checked; parallel |
| `detect` | ingest or a smooth artifact | `detected` | `--on raw/smoothed/residual`; per-chunk reset; 144-sample burn-in |
| `join` | any artifacts + labels | `train`, `test`, `split` | day-level aggregation; patient-level split |
| `model` | a join artifact | `predictions` | `p_id, date, y, y_prob` |
| `evaluate` | predictions or a smooth artifact | `metrics` | picks model-based or signal-based metrics automatically |

`join` aggregates each input artifact to one row per patient-day: numeric columns become
mean/std/min/max, and `*_anomaly` flags become both rows-flagged and **distinct events**
(contiguous runs) — the honest count for a sustained-regime detector like the HMM.

Patients are split, never days: no patient appears in both train and test. The universe
is every patient in the labels file, so a method that drops patients yields fewer rows
but cannot shift the split.

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
consecutive readings ≥ a threshold, and resets every model's state at the boundary.

This constant used to be copied into **13 files** and the function into **10**. Both now
have exactly one definition, and `tests/test_smoke.py` fails if a second appears:

| Where | What |
|---|---|
| `asthma/config.py` | `GAP_THRESHOLD_MIN = 180.0` — the only definition; everything imports it |
| `asthma/chunking.py` | `assign_chunks()` — the only implementation; every smoother imports it |

- **Change the threshold** (e.g. to 120 or 240 min): edit the one line in `config.py`.
- **Change the chunking *strategy***: replace the body of `assign_chunks()` in
  `chunking.py`. It currently emits `chunk_id = (gap ≥ threshold).cumsum()`.
  Alternatives you can drop in: **fixed-length windows**
  (`chunk_id = floor(elapsed_min / W)`), **sleep/wake segmentation** (boundary at
  sleep-stage transitions from the `sleep_active` channel), or **activity-gated**
  (boundary when `steps` spikes). Keep the contract: a monotone integer `chunk_id` per
  row, resetting per patient. Note ids start at **1** — the first row's NaN gap counts
  as a boundary.

**One deliberate inconsistency, pinned by `selftest`:** ingest fills gaps on `> 180`
while the chunker splits on `>= 180`, so a gap of exactly 180.000 min opens a new chunk
but receives no filler row. Preserved from the original Stage 0 because changing it
shifts every downstream output; the 10-min cadence makes it vanishingly rare.

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
controlled by `data/raw/spark/processed_users.txt`.

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

**Logs** (`data/logs/`): `smoothing_<method>_log.csv` (per-patient status,
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

Every pool is discovered from the filesystem, so **an addition is one file**. There is
no registry to edit and no orchestrator to copy.

| To add | Write one file | Entry point it must expose |
|---|---|---|
| a smoother | `wearable/smoothing/<name>.py` | `smooth_dataframe(df, patient_col, timestamp_col, value_col, config, out_col)` |
| a detector | `wearable/cpd/<name>.py` | `detect(time, values, priors) -> {anomaly, phenotype, magnitude, severity}` |
| a model | `wearable/models/<name>.py` | `fit(X, y, seed, **p)` and `predict_proba(fitted, X)` |
| a metric | `wearable/metrics/<name>.py` | `compute(...)` plus `KIND = "model"` or `"signal"` |
| a data format | a `patients()`/`load(pid)` pair in `wearable/sources.py` + one `Source(...)` | — |

Declare `LABEL` and `DEFAULTS` (and `CONFIG` for a smoother) and they show up in
`python -m wearable list`, in every sweep, and in the provenance manifest.

### A smoother must
- `from wearable.chunking import assign_chunks` — **never re-implement it**, and never
  redefine `GAP_THRESHOLD_MIN`. Tests fail if a second definition appears.
- filter each chunk independently with fresh state (H8), run causally (H9), emit a
  strictly positive column (H1), and derive its parameters per patient (H10).
- Check what `selftest` reports about its truncation behaviour. If it is
  truncation-exact, add its key to `TRUNCATION_EXACT` in `wearable/selftest.py` so a
  future regression is caught. If it is not, understand that it is fitting parameters
  on data from the future — see §1 and the H9 note in `CLAUDE.md`.

### A metric must
- be one of the two kinds: `KIND = "model"` takes `(y_true, y_prob)`, `KIND = "signal"`
  takes `(raw, smoothed)` and needs no model at all.
- set `HIGHER_IS_BETTER` (default `True`) so comparison tables rank it correctly.
- return `float("nan")` rather than raising when the input cannot support it (one
  class present, too few points).

### Review checklist
- Never import one step from another; compose in `wearable/experiment.py`. A test
  enforces this.
- Anything that changes a result must be in the artifact key: add it to the step's
  `params`, or results will be wrongly reused. The `--limit` flag is the worked example.
- Prefer data-driven per-patient priors over hard-coded scales (H10).
- Keep determinism: seed any RNG explicitly.
- Run `python -m wearable selftest` before opening a PR.

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
anything under `data/` to a public remote.
