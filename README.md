# Asthma Exacerbation Early-Warning System — HRV Pipeline

Continuous ambulatory **Heart-Rate-Variability** (HRV, 10-minute SDNN) from
consumer smartwatches is processed into an early-warning signal for asthma
exacerbations. The pipeline turns each patient's noisy HRV stream into (a) a
smooth physiological **baseline**, (b) a **residual** that exposes the autonomic
volatility that simple smoothing would erase, and (c) **rich change-point tags**
that become the training target for a downstream ML model.

This README is the operator's manual: what runs, how the files connect, how to
start/stop it, where everything is stored and logged, how every method works and
how its hyper-parameters are chosen, and the full story of the **residual**.

---

## Table of contents

1. [The big idea: the Dual-Track Residual Architecture](#1-the-big-idea-the-dual-track-residual-architecture)
2. [Pipeline at a glance — how the files connect](#2-pipeline-at-a-glance--how-the-files-connect)
3. [Directory & storage layout (where everything lives)](#3-directory--storage-layout-where-everything-lives)
4. [How to run it (start)](#4-how-to-run-it-start)
5. [How to stop it](#5-how-to-stop-it)
6. [Logs — where they go](#6-logs--where-they-go)
7. [The smoothers (Track A): how each runs & how its hyper-parameters are decided](#7-the-smoothers-track-a-how-each-runs--how-its-hyper-parameters-are-decided)
8. [The residual, start to end](#8-the-residual-start-to-end)
9. [The CPD ensemble (Track B): how each method runs & its priors](#9-the-cpd-ensemble-track-b-how-each-method-runs--its-priors)
10. [Output schemas](#10-output-schemas)
11. [Downstream ML & where "monitoring" goes](#11-downstream-ml--where-monitoring-goes)
12. [Dependencies](#12-dependencies)
13. [Troubleshooting & known quirks](#13-troubleshooting--known-quirks)

---

## 1. The big idea: the Dual-Track Residual Architecture

Running change-point detection (CPD) directly on raw HRV fails: it is blinded by
the circadian rhythm and slow baseline drift. Running CPD on *smoothed* HRV also
fails: smoothing erases the high-frequency autonomic volatility that actually
signals an attack. The solution is to use both at once:

```
                 ┌──────────────── Track A (Context) ────────────────┐
   raw HRV ──►  Smoother  ──►  smoothed_hrv   (the expected baseline) │
       │                                                              │
       │                                            residual_hrv  =  raw − smoothed
       │                                                              │
       └──────────────── Track B (Reality) = raw HRV ────────────────┘
                                                              │
                                  CPD ensemble runs on  ►  residual_hrv
                                                              │
                                              Rich Annotation Vector (ML target Y)
```

- **Track A — Context.** A smoothing algorithm estimates the patient's expected
  homeostatic baseline (`smoothed_hrv`).
- **Track B — Reality.** The raw HRV.
- **The intersection — the Residual.** `residual_hrv = raw_hrv − smoothed_hrv`.
  This is what every CPD method analyses. It is mean-zero when the patient is in
  homeostasis and departs from zero exactly when the autonomic system is under
  tension — which is the signal we care about.

All eleven smoothers obey a shared **H-Framework** (positivity, drifting/
multimodal baseline, circadian rhythm, irregular sampling, massive missingness
with a 180-minute segmentation rule, strict causality, and per-patient
adaptation). See `HRV_Smoothing_Methods_Explained.docx` for a plain-language
description of each method.

---

## 2. Pipeline at a glance — how the files connect

The pipeline is a linear chain of numbered stages under `src/`. Each stage reads
the previous stage's output directory and writes its own.

```
raw_data/<pid>/hrv/*.csv
        │
        ▼  src/00_preprocess_raw.py      (parse, gap-fill markers, 180-min rule)
data/processed/<pid>_processed.csv
        │
        ▼  src/01_hyper_tuning.py         (OPTIONAL: particle-filter N*, M*)
data/results/hyperparameter_log.csv
        │
        ▼  src/02*_run_<method>.py        (Track A: 11 smoothers, each its own dir)
data/smoothed/            (02_run_filters.py  — SMC particle filter, FFBS)
data/smoothed_rspf/       (02b — Regime-Switching Particle Filter)
data/smoothed_krlst/      (02c — KRLS-T)
data/smoothed_gpssm/      (02d — State-Space GP)
data/smoothed_ossa/       (02e — Online SSA)
data/smoothed_kim/        (02f — Kim filter)
data/smoothed_gammadglm/  (02g — Gamma CD-DGLM)
data/smoothed_logmhw/     (02h — Log-MHW)
data/smoothed_hrkf/       (02i — Huber Robust KF)
data/smoothed_anrewma/    (02j — AN-REWMA)
        │   each *_smoothed/*.csv now carries  smoothed_hrv  AND  residual_hrv
        ▼  src/03_annotate.py <method>    (Track B: 4 CPD methods on residual_hrv)
data/annotated_<method>/<pid>_..._annotated.csv   (Rich Annotation Vector = ML Y)
        │
        ▼  scripts/plot_<method>.py       (OPTIONAL: per-method static + interactive plots)
data/plots/
```

**Module dependency map**

| Stage | Script | Imports / reads | Writes |
|---|---|---|---|
| 0 Preprocess | `src/00_preprocess_raw.py` | `raw_data/`, `raw_data/processed_users.txt` | `data/processed/` |
| 1 Tune (opt.) | `src/01_hyper_tuning.py` | `data/processed/` | `data/results/hyperparameter_log.csv` |
| 2 Smooth | `src/02_run_filters.py` | `data/processed/` | `data/smoothed/` |
| 2b–2j Smooth | `src/02b…02j_run_*.py` | `data/processed/` + the matching `*_smoother.py` core | `data/smoothed_<m>/` |
| — core | `src/<m>_smoother.py` | (pure library: `smooth_dataframe`, config, self-test) | — |
| 3 Annotate | `src/03_annotate.py` | `data/smoothed_<m>/` + `bocpd.py`,`kcp.py`,`kalman_cpd.py`,`hmm_cpd.py` | `data/annotated_<m>/` |
| Plot (opt.) | `scripts/plot_<m>.py` | `data/smoothed_<m>/` | `data/plots/` |

The orchestrators (`02*_run_*.py`) are the runnable entry points; the
`*_smoother.py` files are importable libraries (each has a `smooth_dataframe()`
and a `__main__` self-test). The four CPD files (`bocpd.py`, `kcp.py`,
`kalman_cpd.py`, `hmm_cpd.py`) are libraries imported by `03_annotate.py`; each
exposes `detect(time, values, priors) -> {anomaly, phenotype, magnitude, severity}`.

---

## 3. Directory & storage layout (where everything lives)

```
asthma-prediction/
├── README.md                         ← this file
├── CLAUDE.md                         ← engineering notes / state-space model spec
├── HRV_Smoothing_Methods_Explained.docx   ← plain-language method guide (for PIs)
├── requirements.txt
├── main.tex                          ← the paper
├── raw_data/
│   ├── "0010"/ hrv/ … , heartrate/ , sleep/ , spo2/ , steps/ , temperature/
│   └── processed_users.txt           ← which patient IDs to process (quoted)
├── src/
│   ├── 00_preprocess_raw.py
│   ├── 01_hyper_tuning.py
│   ├── 02_run_filters.py             ├ orchestrators (runnable)
│   ├── 02b…02j_run_*.py              ┘
│   ├── *_smoother.py                 ← 10 smoother libraries (rspf, krlst, gpssm,
│   │                                    ossa, kim, gamma_dglm, logmhw, hrkf, anrewma)
│   ├── 03_annotate.py                ← CPD ensemble orchestrator
│   ├── bocpd.py / kcp.py / kalman_cpd.py / hmm_cpd.py   ← 4 CPD libraries
│   └── eda.py / eda_smoothed.py      ← diagnostics
├── scripts/
│   ├── plot_<method>.py              ← per-method static PNG + interactive HTML
│   └── visualize*.py
└── data/
    ├── processed/         <pid>_processed.csv        ← Stage 0 output (raw + gaps)
    ├── smoothed/          <pid>_smoothed.csv         ← Stage 2 (particle filter)
    ├── smoothed_<method>/ <pid>_<method>.csv         ← Stage 2b–2j (Track A)
    ├── annotated_<method>/<pid>_<method>_annotated.csv ← Stage 3 (CPD tags = ML Y)
    ├── plots/             <pid>_<method>.png / .html , <pid>_residual_cpd.png
    └── results/           *_log.csv , hyperparameter_log.csv , eda/ , eda_smoothed/
```

**Where each "thing" is stored**

- **Processed (raw + gap markers):** `data/processed/<pid>_processed.csv`
- **Smoothed baseline + residual (Track A):** `data/smoothed/` (particle filter)
  and `data/smoothed_<method>/` (the other ten). Every file contains both
  `smoothed_hrv` and `residual_hrv`.
- **Annotated / "monitored" CPD output (the alarms):**
  `data/annotated_<method>/<pid>_<method>_annotated.csv` — one file per patient,
  carrying the per-method tags and the ensemble Rich Annotation Vector.
- **Plots:** `data/plots/`
- **Logs:** `data/results/` (see §6).

---

## 4. How to run it (start)

All commands run **from the `asthma-prediction/` directory** (so that `./data/`
and `./raw_data/` resolve).

```bash
# 0) one-time: create an environment and install deps
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Stage 0 — preprocess raw Spark CSVs into per-patient series
#    (requires raw_data/processed_users.txt listing patient IDs)
python src/00_preprocess_raw.py

# 2) Stage 1 — OPTIONAL particle-filter hyper-parameter tuning (N*, M*)
python src/01_hyper_tuning.py

# 3) Stage 2 — Track A: run a smoother (pick one or run several to benchmark)
python src/02_run_filters.py        # SMC particle filter      -> data/smoothed/
python src/02b_run_rspf.py          # regime-switching PF       -> data/smoothed_rspf/
python src/02c_run_krlst.py         # KRLS-T                    -> data/smoothed_krlst/
python src/02d_run_gpssm.py         # state-space GP            -> data/smoothed_gpssm/
python src/02e_run_ossa.py          # online SSA               -> data/smoothed_ossa/
python src/02f_run_kim.py           # Kim filter               -> data/smoothed_kim/
python src/02g_run_gammadglm.py     # Gamma CD-DGLM            -> data/smoothed_gammadglm/
python src/02h_run_logmhw.py        # Log-MHW                  -> data/smoothed_logmhw/
python src/02i_run_hrkf.py          # Huber robust KF          -> data/smoothed_hrkf/
python src/02j_run_anrewma.py       # AN-REWMA                 -> data/smoothed_anrewma/

# 4) Stage 3 — Track B: CPD ensemble on residual_hrv for a chosen Track A
python src/03_annotate.py gammadglm   # -> data/annotated_gammadglm/
#    valid method keys: pf, rspf, krlst, gpssm, ossa, kim, gammadglm, logmhw, hrkf, anrewma
#    (default if omitted: gammadglm)

# 5) OPTIONAL — plots for one patient/method
python scripts/plot_gammadglm.py 0010
```

**Run a single module's self-test** (no data needed — synthetic, proves the
H-Framework guarantees of a smoother): `python src/gamma_dglm.py`.

**Parallelism.** Stages 0, 1, and the smoothers (2–2j) use a process pool
(`cpu_count − 1` workers). Stage 3 (annotation) runs patient-by-patient in a
single process.

---

## 5. How to stop it

Every stage is an ordinary foreground Python process, so:

- **Interrupt:** press `Ctrl-C` in the terminal. The orchestrators use a
  `ProcessPoolExecutor`; `Ctrl-C` cancels the pool and stops the run. Partial
  per-patient CSVs already written stay on disk (each patient is written
  atomically as it finishes), so you can re-run and it will simply overwrite.
- **Kill a background run:** if you launched with `&` or `nohup`, find it with
  `ps aux | grep 02g_run_gammadglm` (or the relevant script) and `kill <pid>`;
  use `pkill -f 03_annotate.py` to stop an annotation run.
- **Safe to re-run.** Stages are idempotent — re-running overwrites the matching
  output files; nothing appends. There is no daemon or scheduler to disable.

---

## 6. Logs — where they go

All run logs are CSVs under **`data/results/`**:

| Log file | Written by | Contents |
|---|---|---|
| `phase1_preprocess_log.csv` | Stage 0 | per-patient parse status, row counts |
| `hyperparameter_log.csv` | Stage 1 | recommended particle-filter `N*`, `M*` |
| `smoothing_log.csv` | `02_run_filters.py` | particle-filter per-patient status |
| `smoothing_<method>_log.csv` | `02b…02j` | per-patient status, `n_chunks`, `min_smoothed` |
| `annotation_<method>_log.csv` | `03_annotate.py` | per-patient status, `n_armed`, `n_anomaly`, burn-in `sigma0` |

Each smoother/annotator also prints a live progress bar (`tqdm`) and an
end-of-run summary (success/fail counts, output path, log path) to **stdout**.
`eda/` and `eda_smoothed/` under `data/results/` hold diagnostic EDA outputs.

---

## 7. The smoothers (Track A): how each runs & how its hyper-parameters are decided

**Common run pattern.** Each orchestrator reads `data/processed/<pid>_processed.csv`,
keeps only observed readings (drops the 10-minute NaN gap-fillers so the
180-minute chunker sees the real voids), calls the method's `smooth_dataframe()`
which (i) assigns `chunk_id` at every ≥180-minute gap and (ii) filters each chunk
independently with the state reset at the boundary (causality + segmentation),
then merges `smoothed_hrv` back onto the full grid, computes
`residual_hrv = hrvValue − smoothed_hrv`, and writes the per-patient CSV.

**Two kinds of hyper-parameters.** Every method separates:

- **Per-patient calibration (automatic, data-driven).** Derived from the
  patient's own data, never hand-set — e.g. observed `[min,max]` range, the
  robust noise scale from the MAD of first differences, the log-mean level, the
  median sampling step. This serves H10 (patient variability).
- **Structural constants (global, in the orchestrator).** A handful of knobs that
  control *smoothness/robustness*, set as `GLOBAL_*` at the top of each
  `02*_run_*.py`. These were **tuned empirically on patient 0010's densest
  3-day window** by sweeping each knob and choosing the value that cut the signal
  roughness to ~10–30 % of the raw while still tracking the daily swing. They can
  be overridden per call.

| # | Method | Orchestrator | Structural knobs (current default) | Auto-calibrated per patient |
|---|---|---|---|---|
| 1 | Particle filter (SMC) | `02_run_filters.py` | `GLOBAL_N=250` particles, `GLOBAL_M=50` FFBS paths | latent spreads & σ_obs from MAD, freqs from median dt, bounds from observed [min,max] |
| 2 | RS-PF | `02b_run_rspf.py` | `GLOBAL_K=3` regimes, `GLOBAL_N_PARTICLES=500` | regime means from log-HRV quantiles, noises from MAD |
| 3 | KRLS-T | `02c_run_krlst.py` | `GLOBAL_BUDGET=100`, `GLOBAL_LAMBDA=0.995` (lengthscale=60 min, c=5 in config) | log-mean centering, noise from MAD |
| 4 | State-space GP | `02d_run_gpssm.py` | `GLOBAL_Q_SCALE=0.05`, `GLOBAL_R_SCALE=5.0` | GP-learned transition F, Q/R from residuals, log-mean |
| 5 | OSSA | `02e_run_ossa.py` | `GLOBAL_WINDOW_MAX=144`, `GLOBAL_N_COMPONENTS=2` | log-mean & range |
| 6 | Kim filter | `02f_run_kim.py` | `GLOBAL_Q_SCALE=0.1`, `GLOBAL_R_SCALE=3.0` | regime means from log quantiles, noise from MAD |
| 7 | Gamma CD-DGLM | `02g_run_gammadglm.py` | `GLOBAL_DELTREND=0.99`, `GLOBAL_DELSEAS=0.99`, `GLOBAL_SHAPE_SCALE=1.0` | Gamma shape s = mean²/noise-var, prior level = −log(median) |
| 8 | Log-MHW | `02h_run_logmhw.py` | `GLOBAL_ALPHA=0.08`, `GLOBAL_GAMMA=0.02`, `GLOBAL_OPTIMIZE=True` | per-patient circadian profile; α/β/φ MLE-fit via statsmodels |
| 9 | H-RKF | `02i_run_hrkf.py` | `GLOBAL_DELTREND=0.99`, `GLOBAL_DELSEAS=0.99`, `GLOBAL_HUBER_DELTA=1.345` | measurement noise R from MAD, prior level from median |
| 10 | AN-REWMA | `02j_run_anrewma.py` | `GLOBAL_NORM_ALPHA=0.05`, `GLOBAL_EWMA_C=1.0` | rolling mean/std init from data, observed range |

> The original SMC particle filter (`data/smoothed/`) uses FFBS backward
> smoothing, which looks at future samples — its `smoothed_hrv` (and therefore its
> `residual_hrv`) is **non-causal** and is kept for offline comparison only. The
> ten causal smoothers (2b–2j) are the ones intended for the real-time warning
> system.

To re-tune any method, change its `GLOBAL_*` constants (or pass overrides to the
`run_*` function) and re-run that one orchestrator; nothing else is affected.

---

## 8. The residual, start to end

This is the spine of the whole system; here is its full life-cycle.

1. **Birth — raw vs. baseline.** After Track A produces `smoothed_hrv` for a
   chunk, the orchestrator computes, on every observed row,
   `residual_hrv = hrvValue − smoothed_hrv`. (On gap-filler rows there is no raw
   value, so the residual is `NaN`.) This single line lives in every orchestrator
   and in every `smoother.smooth_dataframe()`.
2. **Meaning.** Because Track A is the *expected homeostatic baseline*, the
   residual is the part of HRV the body's normal physiology does **not** explain.
   In homeostasis it hovers around 0; under autonomic tension it deviates. It is
   the "current tension" channel, complementing the "circadian context" channel
   (`smoothed_hrv`).
3. **Causality.** For all ten causal smoothers, `smoothed_hrv` uses only
   past/present data, so `residual_hrv` is also strictly causal — usable in a
   real-time monitor.
4. **Calibration (24-hour burn-in).** In Stage 3, the first **144 observed
   residual points** (≈24 h) of each patient set the noise priors: the median
   `m0` and a robust std `σ0` (`1.4826 × MAD`, floored). These feed BOCPD's
   variance prior, the Kalman measurement noise `R`, and the HMM emission
   variances. During this burn-in the ensemble is **not armed** (alarms
   suppressed, phenotype `burn_in`); after 144 points it arms.
5. **Detection.** All four CPD methods run **on `residual_hrv`** — never on raw
   or smoothed — independently per `chunk_id` (state resets at each ≥180-min gap).
6. **Output.** Per-method and ensemble tags are written next to the residual in
   `data/annotated_<method>/`. The residual column itself is carried through so
   downstream ML can use it directly as a feature.

A verification plot of this for patient 0010 (Track A = Gamma DGLM) is
`data/plots/0010_residual_cpd.png`: top panel shows raw + baseline; bottom panel
shows the residual oscillating around 0 with ensemble anomalies coloured by
phenotype.

---

## 9. The CPD ensemble (Track B): how each method runs & its priors

`src/03_annotate.py` orchestrates four online detectors over `residual_hrv`,
per patient, per chunk, after the 24-hour burn-in. Each `detect()` returns a
**Rich Annotation Vector** per timestep — `anomaly` (0/1), `phenotype` (string),
`magnitude` (native score), plus an internal `severity` (normalised, used only to
pick the primary phenotype). The hyper-parameters below come from the
architecture spec (clinical priors), not from data sweeps; only the noise scales
are data-driven via the burn-in.

### BOCPD — `bocpd.py` → `BOCPD_Mean_Shift`
- **What it does:** Bayesian Online Change-Point Detection (Normal-Normal
  conjugate) maintains a posterior over the current "run length" (time since the
  last change). At a sustained mean shift the run-length posterior collapses.
- **Priors:** prior mean `μ0 = 0` (the residual's normal state); hazard
  `λ = 1/1000` (≈ one severe exacerbation per ~1000 ten-minute points ≈ a week);
  prior variance from the burn-in `σ0`.
- **Trigger:** a **run-length collapse** — the most-probable run length, after
  having grown ≥ `MIN_RUN=6` (~1 h), drops below `DROP_FRAC=0.5` of its prior
  value. *(Note: with hazard 1/1000 the raw `P(run-length=0)` can never reach the
  literal 0.95 the spec names, so we use the spec's equivalent "run-length drop"
  criterion.)* **Magnitude** = the normalised run-length drop.

### KCPD — `kcp.py` → `KCPD_Distribution_Shift`
- **What it does:** a causal, online two-window **Maximum Mean Discrepancy
  (MMD)** test — compares the distribution of the *past* window to the *present*
  window, so it catches non-parametric shape/skewness changes, not just means.
- **Priors:** window `w = 18` (3 h per side); RBF-kernel bandwidth by the
  **median heuristic** (adapts per chunk); evaluated every `STRIDE=3` points.
- **Recalibration (important).** A naive "fire if permutation p<0.05" over-fires
  badly: (1) circadian heteroscedasticity leaks into the residual (night
  volatility > day), so the test flags normal day↔night variance transitions;
  (2) significance ≠ relevance — with 36-point windows nearly everything is
  "significant," and one test per step is a huge multiple-comparison problem.
  Two fixes: the residual is **studentized** by a causal ~6h rolling std
  (`STUD_HALFLIFE=36`) so only departures *beyond* the normal volatility envelope
  register; and the trigger is an **effect-size gate** — the MMD must sit
  `MIN_EFFECT_Z=10` null-standard-deviations above its own permutation null
  (`N_PERM=80`), a multiplicity-robust criterion. The anomaly is marked at the
  single **event point** (not held over the stride), so rows = events.
- **Trigger:** MMD effect-size `z ≥ 10`. **Magnitude** = the MMD² distance.
- **Effect of the fix (patient 0010):** KCPD dropped from 317 events (3223
  stride-held rows) to **125 events** — in line with BOCPD (109), Kalman (147),
  HMM (108) — and its firing rate stopped tracking time-of-day (was 0.12–0.41 by
  block, now 0.004–0.023).

### Kalman innovations — `kalman_cpd.py` → `Kalman_Trend_Break` / `Kalman_Volatility_Spike`
- **What it does:** because Track A already removed the baseline, this Kalman
  filter tracks the residual's `[level, slope]` (expected slope 0) and watches
  its one-step prediction errors (innovations).
- **Priors:** measurement noise `R = (R_SCALE=1.0 × σ0)²` (kept *high relative to*
  the low process noise `Q_FACTOR=1e-3`, so innovations are standardised ~N(0,1)
  under normal data); a two-sided **CUSUM** of standardised innovations with
  slack `k=0.5` and threshold `h=3`.
- **Trigger:** a single `|z| > 3` → `Kalman_Volatility_Spike`; a CUSUM breach
  (sustained one-directional drift, e.g. HRV creeping down for hours without a
  sudden drop) → `Kalman_Trend_Break`. **Magnitude** = the innovation Z-score.

### HMM — `hmm_cpd.py` → `HMM_State_Transition`
- **What it does:** a 2-state Gaussian HMM that decides (via Viterbi) whether the
  patient is in **State 0 = Homeostasis** or **State 1 = Pathological**.
- **Priors:** sticky transition matrix (diagonal `0.95`, off-diagonal `0.05`) — a
  shock-absorber against flip-flopping; emissions **hard-coded**: both states
  mean 0, State 0 variance `= σ0²`, State 1 variance `= 9·σ0²` (high volatility).
  Only the transition matrix is learned (`params="t"`); the emissions stay fixed,
  so State 1 remains a genuine high-variance regime.
- **Trigger:** Viterbi decodes State 1. **Magnitude** = `P(State 1)`.

### Ensemble (the ML target Y)
- Per-method columns `{m}_anomaly`, `{m}_phenotype`, `{m}_magnitude` are written
  for `m ∈ {bocpd, kcpd, kalman, hmm}`.
- **`anomaly_present` = ANY of the four fires** (after arming) — the most
  sensitive vote, chosen deliberately; false alarms are meant to be tuned down
  later by the Human-in-the-Loop loop, which lowers the voting weight of whichever
  method over-fires for a given patient.
- **`primary_phenotype`** = the firing method with the highest normalised
  severity; **`magnitude`** = that method's native magnitude.
- During burn-in everything is `burn_in` / 0.

> **Observed behaviour (patient 0010, Track A = Gamma DGLM), after KCPD
> recalibration:** distinct events are now comparable across methods — BOCPD 109,
> KCPD 125, Kalman 147, HMM 108 — and the ensemble (any-of-4) is 352 events. The
> primary-phenotype mix is diverse (HMM, KCPD, Kalman trend/spike, BOCPD), rather
> than KCPD dominating. Compare methods by **events**, not flagged rows (HMM in
> particular marks every row of a volatility regime, so its row count is high
> while its event count is low).

---

## 10. Output schemas

**Smoothed file** — `data/smoothed_<method>/<pid>_<method>.csv`:

```
createdTime, hrvValue, minute_diff, smoothed_hrv, residual_hrv, gap_flag, chunk_id
```
(`02b_run_rspf.py` and `02_run_filters.py` additionally carry `true_trend_level`.)

- `smoothed_hrv` — Track A baseline (>0).
- `residual_hrv` — `hrvValue − smoothed_hrv` (NaN on gap-filler rows).
- `gap_flag` — 1 on inserted gap-filler rows, else 0.
- `chunk_id` — segment id, incremented at every ≥180-min gap.

**Annotated file** — `data/annotated_<method>/<pid>_<method>_annotated.csv`:
the smoothed columns above **plus**, for each `m ∈ {bocpd, kcpd, kalman, hmm}`:

```
{m}_anomaly, {m}_phenotype, {m}_magnitude
```
**plus** the ensemble Rich Annotation Vector and provenance:

```
anomaly_present, primary_phenotype, magnitude, cpd_input_col(=residual_hrv)
```

---

## 11. Downstream ML & where "monitoring" goes

There is **no live daemon yet** — the pipeline is batch. The "monitoring output"
(the alarms) is the annotated CSV in `data/annotated_<method>/`. That feeds the
planned ML stage:

- **ML input matrix X (per timestep):** `smoothed_hrv` (circadian context),
  `residual_hrv` (current autonomic tension), `steps`/`sleep` (exogenous
  behaviour), and `time_of_day` (sin/cos encoded).
- **ML target Y:** the **Rich Annotation Vector** (`anomaly_present`,
  `primary_phenotype`, `magnitude`) — multi-task: predict the *phenotype* and
  *severity*, not just a binary flag.
- **Self-supervised pre-training (SSL):** mask 4-hour windows of `residual_hrv`
  in healthy chunks and train the model to reconstruct them, so it learns
  physiology before seeing CPD tags.
- **Human-in-the-Loop (HITL):** patient-reported outcomes adjust Track A
  "stiffness" (the smoother's `GLOBAL_*` knobs) and the per-method voting weights
  in Track B to suppress that patient's false alarms.

When the live monitor is built, it will run the same causal smoother + CPD
ensemble incrementally and write alarms to a monitoring store; the file contract
in §10 is what it will emit.

---

## 12. Dependencies

Developed on **Python 3.10**. Install with `pip install -r requirements.txt`.
Key packages: `numpy`, `pandas`, `scipy`, `scikit-learn` (GP kernels, HMM
preprocessing), `statsmodels` (Holt-Winters fitting), `hmmlearn` (HMM CPD),
`ruptures` (legacy kernel CPD; the current `kcp.py` is self-contained),
`particles` (SMC), `matplotlib` + `plotly` (plots), `tqdm`.

---

## 13. Troubleshooting & known quirks

- **"No residual_hrv column" in Stage 3** — re-run the matching smoother; the
  residual is added by the Stage 2 orchestrators.
- **FFBS particle filter residual is non-causal** — use a causal smoother
  (2b–2j) as Track A for anything real-time.
- **KCPD tuning** — it studentizes the residual (causal ~6h rolling std) and
  fires on an MMD effect-size gate `MIN_EFFECT_Z=10` (not a raw p<0.05), which
  keeps it in line with the other detectors and removes the circadian-artifact
  over-firing. Raise `MIN_EFFECT_Z` for fewer alarms, lower it for more; set
  `STUDENTIZE=False` to test on the raw residual.
- **"Model is not converging" from hmmlearn** — harmless on short chunks; the
  logger is silenced in `hmm_cpd.py` and the chunk still returns a valid result.
- **A chunk shorter than a method's minimum** (`MIN_SEGMENT_ROWS=10`, or 2×window
  for KCPD) is left unsmoothed/unflagged by design.
- **Re-running is safe** — every stage overwrites its per-patient outputs.
```
