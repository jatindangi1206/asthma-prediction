# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research pipeline for wearable time-series work: turn noisy, gap-ridden ambulatory
signals (10-min SDNN HRV from consumer smartwatches) into supervised early-warning
signals for respiratory exacerbations. Asthma is the first study, not the repo's
identity — the cohorts are AIIMS asthma adults/children, COPD and ACHE.

`README.md` is the full manual; where it, this file and the code disagree, the code
wins (e.g. the README's `python -m wearable all ...` does not exist — see
`wearable/__main__.py`). `main.tex` is the paper. **Run everything from the
`asthma-prediction/` directory** — `config.DATA` is the relative `./data`.

## The five rules this repo is built around

These are not aspirations; `python -m wearable selftest` and `tests/test_smoke.py`
assert them, and a PR that breaks one fails.

1. **Steps do not depend on each other.** Only `wearable/experiment.py` connects them.
   A test greps for cross-step imports.
2. **Adding a method is one file, nothing else.** Pools are discovered from the
   filesystem (`wearable/registry.py`). There is no list to register in.
3. **Every result records how it was made** — method, settings, input artifacts, code
   version — in a `manifest.json` beside it. Any result traces back to the raw signal.
4. **The same computation is never run twice.** Artifacts are keyed on a content hash;
   a matching key is reused. Editing a method invalidates only that method — but read
   "What the cache key does not see" below before trusting a reuse.
5. **A comparison varies only the method.** Same test participants, same model, same
   seed. The split is derived from the patient universe and the seed alone.

## Layout

```
wearable/
  config.py        all constants, paths, seed, store format   ← the only definitions
  chunking.py      assign_chunks() — the H8 rule              ← the only implementation
  registry.py      filesystem discovery for all four pools
  store.py         artifacts: provenance manifests, content-keyed reuse, format setting
  sources.py       data formats      patients() + load(pid)
  windowing.py     the window rule — the only implementation (join + encode)
  smoothing/       10 methods        smooth_dataframe(...)
  forecasting/     3 methods         forecast(contexts, horizon) -> (median, quantiles)
  encoders/        2 methods         encode(series) -> (n, d)
  cpd/             5 methods         detect(time, values, priors) + score(); _whiten.py
  models/          10 models         fit(...) + predict_proba(...)
  metrics/         19 metrics        compute(...) + KIND
  steps/           ingest · smooth · forecast · detect · encode · join · model · evaluate
  experiment.py    the ONLY thing that connects steps: chain() and the sweeps
  selftest.py      asserts the five rules
  __main__.py      the CLI
scripts/           reporting: eda, plot_patient, smoother_comparison_report, smoothing_study
legacy/            retired code (see legacy/README.md)
docs/              method notes, the PDF and DOCX
tests/             smoke tests — enforce rules 1, 2, 4, 5
data/              see the naming rule below; never committed
  raw/{pass1,spark}/   inputs, never written by the pipeline
  processed/           ingest output + labels_pass1.csv
    <pid>_processed.csv            hrv (the legacy name, unchanged)
    <pid>__<channel>_processed.csv hr · temp · spo2 · steps
  runs/                the artifact store — current results + provenance
  reports/{eda,plots,smoother_comparison}/   regenerable human-facing output
  logs/                per-step run logs
  old/                 everything from previous runs; never read by the pipeline
```

**Naming rule.** Lowercase `snake_case` everywhere. Top-level directories are named by
kind (code / prose / data); everything under `data/` is named by lifecycle stage, so the
name says which stage produced it and whether it is current — `runs/` current and
provenance-tracked, `reports/` regenerable, `old/` previous runs. Artifact filenames
are `<pid>__<step>_<method>.<ext>`.

Flow (steps compose in any order; none is mandatory):

```
data/raw/pass1/ or data/raw/spark/
  └─ ingest  → data/processed/<pid>_processed.csv + data/processed/labels_pass1.csv
       ├─ smooth  <method>          → table `smoothed`  (adds residual_hrv)
       ├─ detect  --on raw|smoothed|residual → table `detected`
       └─ join    any artifacts + labels    → tables `train`, `test`
            └─ model   → table `predictions`
                 └─ evaluate → table `metrics`
```

Everything from `smooth` on writes to `data/runs/<step>/<method>/<key>/`.

## Commands

```bash
# Python 3.10 venv; `python` below means .venv/bin/python. Call it directly — `source
# .venv/bin/activate` does not always take in a non-interactive shell, and any python
# outside the venv breaks on numpy (x86_64 ImportError on Apple Silicon; "numpy.dtype
# size changed" with anaconda's 3.12 on Linux).
python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt
# On the Linux workstation the system python3.10 has no ensurepip, so .venv is a conda
# env (`conda create -p .venv python=3.10`), and ~/.local site-packages leak into it:
# export PYTHONNOUSERSITE=1 before every install and run.

python -m wearable list                   # every discovered method, with its defaults
python -m wearable selftest               # assert the five rules; touches no data
python -m wearable runs [step] [method]   # the artifact log
python -m wearable lineage smooth:gammadglm   # trace a result back to the raw signal

python -m wearable ingest [--source pass1|spark] [--channels hrv,hr]
python -m wearable smooth gammadglm [--signal hrv|hr|temp|steps] [--deltrend 0.995] [--force]
python -m wearable detect --on residual --smoothing gammadglm --signal hr
python -m wearable forecast timestral --signal hr   # frozen TSFM -> surprise + tail
python -m wearable encode moment --signal hr --days 7 --profile 73
python -m wearable detect --on surprise --forecasting timestral --signal hr
python -m wearable detect --on raw            # CPD straight on the raw signal
python -m wearable detect --on residual --smoothing gammadglm --methods bocpd,kcpd
python -m wearable join --artifacts smooth:gammadglm,detect:bocpd --target detection
python -m wearable model join:smooth:gammadglm --model logreg
python -m wearable evaluate model:logreg

python -m wearable chain --smoothing gammadglm --cpd bocpd,kcpd --on residual
python -m wearable chain                      # no smoothing, no CPD: raw baseline

python -m wearable sweep smoothing            # every smoother + baseline, one model
python -m wearable sweep cpd                  # every detector, same signal + model
python -m wearable sweep signal               # one ensemble on raw vs smoothed vs residual
python -m wearable sweep smoothers-no-model   # judge smoothers with no model at all
python -m wearable sweep inputs --task detection|onset   # the modelling experiment
python -m wearable sweep models --variant residual+cpd   # same grid, model as axis
python -m wearable sweep window --task detection         # 7 days of input down to 1
python -m wearable sweep window --task onset --horizon 1 # ... asking about tomorrow
python -m wearable join --days 7 --target onset --horizon 1   # one dataset by hand
python -m wearable sweep smoothing --limit 40 # first 40 patients; limit is in the key
```

`smooth` and `sweep` take `--<knob> <value>` pairs parsed as Python literals and passed
straight through as kwargs (so `--limit` works there; `chain` has no `--limit`).

**Artifact specs** are `step:method[:key]` with the split at the *first* colon (method
names contain colons: `join:smooth:gammadglm`, `evaluate:model:logreg`). Without a key
the newest complete artifact of that name wins. Detect artifacts are named by the
ensemble (`detect:bocpd+kcpd+kalman+hmm+drift`), not by `--on`/`--smoothing`, so runs on
different signals share a name — pass the key from `runs` when that matters. Join
artifacts are the same: the name is the input combination, not `--target`/`--days`/
`--horizon`, so every window and both tasks share one name. The cache keys them
correctly; it is only bare-name resolution that picks the newest.

Tests: `python -m wearable selftest` (same checks as `tests/test_smoke.py`, readable
output, no data needed). `pytest` is **not** in the venv; run one test
directly:

```bash
.venv/bin/python -c "from tests.test_smoke import test_steps_do_not_import_each_other as t; t()"
```

**Smoothing study** (`scripts/smoothing_study.py stats|derive|run|report|all`): derives
every smoother's settings from the cohort being smoothed (AIIMS, COPD, combined) via
`wearable/cohort_stats.py` — a level-plus-white-noise variogram (R, S, tau), cadence,
circadian, volatility and activity-state statistics — runs defaults vs derived, scores
per patient, and writes `data/reports/smoothing/` (settings, metrics, `methods.html`,
and `dashboard/index.html`, which loads `patients/<pid>.js` so it works from file://).
`smooth.run(..., patients=[...])` restricts a run to a cohort; the id list is keyed.
The two HTML pages are **templates in `scripts/`** (`smoothing_dashboard.html`,
`smoothing_methods.html`): `report` copies the dashboard and injects data into the
methods page at the `/*__DATA__*/null` placeholder. Edit the templates, not the
generated copies under `data/reports/smoothing/`.

## What the cache key does not see

`key = sha256(step, method, params, input keys, code_hash)`, where `code_hash` is the
**method module's file hash** only. So a reuse can be stale when you change:

- **step code** (`wearable/steps/*.py`), `chunking.py`, or constants in `config.py`
  (`GAP_THRESHOLD_MIN`, `MIN_SEGMENT_ROWS`, …) — none are hashed. Every smoother imports
  `chunking`/`config`, yet editing them invalidates nothing.
- **evaluate's scoring logic** — a predictions artifact's `code_hash` carries
  `steps/evaluate.py`'s `PRED_LOGIC` (`"eval2"`); bump it when the scoring output
  changes without a metric file changing. Signal scoring keeps its original key on
  purpose, so stored smoothing results stay reusable.
- **join logic** — join's `code_hash` is the literal `"join3"`; bump it in
  `steps/join.py` when join behaviour changes.
- **ingest output** — smooth and `detect --on raw` key the ingest input as the *path
  string* `data/processed`, and `labels_pass1.csv` is not keyed at all. Re-ingesting
  does not invalidate anything downstream.

In all three cases pass `--force` (it cascades through `chain`), or delete the stale
`data/runs/<step>/` subtree.

## Adding things — one file each

| To add | Write | Then |
|---|---|---|
| a data format | `patients()` + `load(pid)` in `wearable/sources.py` + one `Source(...)` | every step unchanged |
| a signal channel | one entry in `config.CHANNELS` | ingest writes it; `--signal` smooths and detects it |
| a smoother | `wearable/smoothing/<name>.py` with `smooth_dataframe` | nothing — it is discovered |
| a detector | `wearable/cpd/<name>.py` with `detect` (+ `score` or `threshold_for` to be calibrated) | add the key to `DEFAULT_ENSEMBLE` to put it in the vote |
| a model | `wearable/models/<name>.py` with `fit` + `predict_proba` | nothing |
| a metric | `wearable/metrics/<name>.py` with `compute` + `KIND` | nothing — it appears in every comparison |
| an input variant | one entry in `experiment.INPUTS` | nothing — `sweep inputs` picks it up |

Optional module attributes: `LABEL`, `DEFAULTS` (the knobs — these replaced the old
`GLOBAL_*` constants), `CONFIG` (the dataclass), `KIND` (metrics: `"model"` takes
`y_true, y_prob`; `"signal"` takes `raw, smoothed`), `HIGHER_IS_BETTER`.

A module whose name starts with `_`, or which lacks the entry point, is skipped — so
helpers can live in a pool directory. A method that fails to import is reported as
unavailable rather than taking the registry down — but `selftest` fails on any
unavailable method, so a new dependency must go in `requirements.txt`.

A new smoother must emit a strictly positive baseline (H1, asserted). If it is
truncation-exact (H9, below) — verify at a few hundred readings, not a few dozen, or a
whole-frame statistic will not have moved yet — add it to `TRUNCATION_EXACT` in
`wearable/selftest.py`;
selftest then fails if it ever regresses.

## Signal choice: raw, smoothed, or residual

`detect --on` picks the column. `residual_hrv = hrvValue − smoothed_hrv` is computed by
the smooth step because it is a property of smoothing, **not** the architecture it used
to be. The research rationale still holds — CPD on raw false-alarms on circadian drift,
CPD on smoothed erases the volatility bursts that signal an attack — but it is now one
option of three and `sweep signal` measures whether it actually wins.

## The modelling experiment

Ported from the COPD repo's `exac/` design (its `docs/EXAC_DESIGN.md`), re-expressed in
this repo's pools and artifact store. **The experimental logic is reused; none of its
code is.**

- **Two tasks, kept separate and never in one table.** `detection` = `label` on `sample`
  days; `onset` = `onset_label` on `onset_valid` days (a new episode within 1–3 days).
  `join.TARGETS` maps each to (label, validity, event id) and already existed.
- **Three axes, all of them `join`'s, none of them a model's.** A model is handed
  columns and never learns which axis produced them, so every axis sweeps the whole
  pool unchanged.
  - *input* — `experiment.INPUTS` = raw · smoothed · raw+cpd · smoothed+cpd ·
    residual+cpd, each just a recipe for the artifact list `join` receives. A variant
    is one dict entry.
  - *window* (`--days`, `config.WINDOW_DAYS`) — the classified day plus the `days-1`
    before it, as `<feature>_d0 … _d<N-1>` lag columns. `days=1` is the classified day
    alone and returns the pre-window dataset untouched. A day missing from the
    recording gives NaN, never the next-oldest day shifted into its place — which is
    why `_window` merges on a forward-dated copy instead of calling `shift(k)`.
  - *horizon* (`--horizon`, onset only, `config.WINDOW_HORIZON`) — how far ahead onset
    is asked about. The labels hold "a new episode within 1–3 days"; a smaller horizon
    asks a stricter question of **the same days**, clearing the label and the episode
    id of positives whose episode starts later. The day set never moves, so two
    horizons share a denominator and can be read side by side.
- **Folds:** `config.TEST_PCT` (20/10/30/40/50 % test) × `config.SEEDS` (0–4) = 25. The
  seed drives the split *and* the model.
- **The split is stratified** by cohort group × has-any-exacerbation (`patient_split`'s
  `strata` argument), because an unstratified draw can leave a fold with no positive
  test patient — the paediatric group has 8 exacerbation days in total. Strata and
  universe come from the **detection** labels whatever the task is, so both tasks score
  the same test patients. This trades a little of the hash split's growth stability;
  `stratify=False` restores it.
- **Scoring is three-scoped:** per day (11 `KIND="model"` metrics at
  `config.THRESHOLD`), per episode (`KIND="event"`, handed the whole predictions frame
  because an episode id is not recoverable from `(y_true, y_prob)`), and per
  probability band (the `calibration` table). `lo`/`hi` resample **test patients**, not
  days — resampling days reports an interval several times too narrow. Only
  `config.BOOT_METRICS` get one.
- **Fairness is asserted, not assumed.** `_grid` raises if two cells in a fold were
  scored on different test days, positives, episodes or patients. The fold key includes
  the horizon (a narrower question legitimately has fewer positives) but **not** the
  window — so the same assertion proves that going from 1 to 7 days of input did not
  quietly change what was being scored.
- `always_no` is a model like any other, so accuracy is readable at a 1 % base rate.
- `data/reports/experiments/runs_<task>.csv` is read, merged and **rewritten** across
  runs (appending corrupts it when a run's column set differs, e.g. `--boot 0`); a
  re-run of the same cell replaces the earlier one. `summary_<task>.csv` is rewritten.
  A cell is keyed by `experiment.CELL` — which **includes `days` and `horizon`**, or a
  3-day run would overwrite a 7-day run's row.

**The model pool (7).** `always_no` (reference), `logreg`, `histgb`, `xgboost`,
`catboost`, `rnn`, `lstm`. The tree models carry the reference design's own settings
(400 trees, depth 4, lr 0.05) and weight the positive class. `rnn`/`lstm` read the
window as a sequence, recovering it from the `_d<k>` column names alone via
`models/_seq.py` — so they need no new contract between the steps, and they run on any
window including `days=1` (one time step). Their input is the scaled values **plus a
"day present" flag per feature**, as the reference fed its networks readings plus
presence masks.

`xgboost`, `catboost` and `torch` are in `requirements.txt` — **`selftest` fails on a
method it cannot import**, so the model files and the install go together.

Two deliberate departures from the reference, both forced by the input contract:

- **No early stopping in `rnn`/`lstm`.** The reference held back *patients* to
  early-stop on; this repo's model contract is handed features and a label and never
  sees `p_id`, so a holdout here would put the same patient on both sides and
  early-stop on leaked information. The step count is fixed instead
  (`_seq.DEFAULTS["steps"]`); every other part of the loop is the reference's
  (half-exacerbation batches, Adam, BCE on the logit, gradients clipped).
- **GAST and TimesFM are not ported.** GAST is one token per *minute* with a reading,
  with state decaying by the real elapsed time between tokens — at day resolution its
  gap-aware machinery degenerates to the day-present mask `_seq.py` already carries, so
  porting it would be `lstm` under another name. TimesFM needs its downloaded
  checkpoint and carries a non-commercial licence; it is one file when that is in place.

Sizing note, not a bug: a 7-day window is 7 time steps and the cohort has 144
exacerbation days. The reference ran its networks on 10,080 × 7 values per sample, where
a recurrent encoder has something to compress. Here they are a fair row in the
comparison, not the favourite — read them next to `histgb` and `xgboost`.

## H-Framework

H1 positivity, H2/H4 drifting multimodal baseline, H5 circadian, H7 irregular sampling,
**H8 ≥180-min segmentation (state flush)**, **H9 strict causality**, H10 per-patient
adaptation.

**H9 is weaker than it looks — do not claim it repo-wide in the paper.** `selftest`
probes it by truncation (does a prefix run reproduce the full run where they overlap?).
Of the ten hand-built smoothers **exactly one — `anrewma` — is truncation-exact.** The
other nine filter forward but fit their per-patient parameters (H10: MAD-noise,
log-mean, observed range, median dt) on the *whole frame handed in*, so a value at time
t moves when later rows arrive. That is a lookahead path, and a residual computed with
parameters that saw the future is an optimistically biased early-warning signal.

Deviations at the current probe (200 readings, 140 overlapping):

```
rspf 3.16  logmhw 1.73  ossa 0.739  hrkf 0.357  gpssm 0.326
krlst 0.270  gammadglm 0.127  kim 0.115  ctimm 0.097  anrewma 0
```

**`ctimm` was listed as exact until 30 Oct 2026 and is not.** The old probe was 60
readings with 40 overlapping — too short for either of the whole-frame statistics it
fits (the observed log range and a MAD noise variance `R`, `ctimm.py:162-164`) to come
out different, so it passed. Its own module demo cannot catch this either: it computes
`R` once and hands the same value to both runs, testing the filter's forward-only-ness
given fixed parameters rather than truncation exactness end to end. The probe is now
200/140. Treat the older numbers in this file and in `README.md` as measured at a length
that flattered every method — `hrkf` in particular read 0.002 and is really 0.357.

To close it, derive those parameters from a causal expanding window. The foundation
models sidestep it entirely — frozen weights, nothing fitted on the cohort, and the
classified day is never in their context — but they are in the **models** pool, so they
do not give you a leak-free residual for CPD. See "Foundation models" below.

## Foundation models

`timestral` (Mistral Timestral-209M, Modified MIT) and `timesfm` (Google TimesFM) are
frozen zero-shot forecasters in the **models** pool. Shared machinery:
`models/_foundation.py`; the adapters are ~30 lines each.

**They need the day, not statistics about it.** A modelling table normally carries four
summary numbers per day, which is nothing for a model trained on sequences. So they read
the profile columns `join --profile 144` emits — `<prefix>_p<slot>[_d<k>]`, recovered
from the column names alone, as `_seq.py` does for `rnn`/`lstm`. At
`--days 7 --profile 144` that is a 1,008-point series per row at the native 10-minute
cadence.

**How a forecaster becomes a classifier:**

```
context = the window's earlier days (k >= 1)   ->  frozen model forecasts the day
target  = the classified day itself (k == 0)   ->  surprise = target - forecast
                                               ->  logistic head trained on 7 features
```

`err_mean · err_abs_mean · err_abs_max · err_sd · share_below · n_obs_target ·
n_obs_context`. The heavy model is frozen and the head is trained — with 144
exacerbation days that is the only split of labour that is not immediate overfitting,
and it is what the reference design did (its 6.8: "frozen, + a learned logistic layer").

- **Strictly causal**: the forecast never sees the classified day, because that day is
  what is being predicted. Nothing is fitted on the cohort either.
- **The head sees surprise only**, not the day features, so the row answers "what does
  frozen-foundation-model surprise achieve on its own" and is comparable against
  `logreg` on day features. `--model_params "{'with_day_features': True}"` hands it both and asks
  whether surprise *adds* anything.
- **A 1-day window is rejected**: with no earlier day there is nothing to forecast from.
- **The frozen pass is cached on disk** (`data/runs/_foundation_cache/`) keyed by each
  row's context bytes, deduped within a call as well. It does not depend on the fold,
  the seed or the head, so five seeds cost one pass. Delete the directory to redo it.
- **Only the median is used**, so the surprise is a plain signed error. The 21 quantiles
  would give a calibrated tail probability instead — a better-posed surprise, one more
  function in `_foundation.surprise`, not built.
- **Both are imported lazily and are NOT in `requirements.txt`**, deliberately
  departing from the "a new dependency must go in requirements.txt" rule: an
  unimportable method fails `selftest`, and licence-gated multi-gigabyte weights have no
  business in an architecture check. The registry lists them; a missing package raises
  an actionable error on first use.
- TimesFM's default checkpoint is **2.5** (Apache-2.0). The 3.0 weights are
  non-commercial and its Python API moved between releases — published examples
  disagree on the entry point — so 3.0 is not the default and its forecast signature is
  unverified here. `_forecast` is the only place the vendor API is touched.
- Timestral-209M reportedly beats TimesFM 3 on the public benchmarks and is Modified
  MIT: prefer it unless you want the comparison.

```bash
python -m wearable sweep models --variant raw --days 7 --profile 144 \
    --model_names timestral,logreg,histgb,presence_only,always_no \
    --test_pct 20 --seeds 0,1,2,3,4
```

## Channels: the pipeline is multi-channel

`config.CHANNELS` maps each channel to its storage column and is the only definition.
Ten channels, in two groups:

| group | channels | smoothable | why |
|---|---|---|---|
| continuous | `hrv` (stored as `hrvValue`, the legacy name), `hr`, `temp`, `spo2`, `steps` | **yes** (`config.SMOOTHABLE`) | a level worth estimating |
| sleep & activity | `sleep_active`, `steps_active` (0/1 on ~100% of rows), `deep_sleep`, `light_sleep`, `almost_awake` (~0.3%, 5-minute durations including 0) | **no** | a flag has no level to estimate, and H1 demands a strictly positive baseline |

`smooth`, `forecast` and `encode` raise on a non-smoothable channel and name the fix.
Sleep still reaches the model: `join --channels hrv,hr,sleep_active` reads one feature
block per channel, so "hours asleep", "minutes of deep sleep" and sleep fragmentation
become day features. A state flag is never profiled — 144 copies of a 0/1 flag say
nothing a daily mean does not. `channels` enters join's key only when it is not the
single-channel default, so existing join artifacts keep their keys.

**Nothing in a pool was ever HRV-specific.** `smooth_dataframe` always took `value_col`
as an argument, so `smooth gammadglm --signal hr` runs the same ten smoothers over heart
rate with **zero changes to any method**. Only the steps had HRV written into them.

**One file per patient-channel, not one file holding every channel.** The channels differ
enormously in cadence and H8 segments a series on *its own* observation times, so a
shared grid would hide one channel's gaps behind another's readings. Each channel gets
its own grid, its own filler rows and its own `minute_diff`.

Measured coverage (40 patients, 2,167,528 raw minute rows; segments over 20 patients):

| channel | share of raw rows | obs (median) | per unbroken segment | per day |
|---|---|---|---|---|
| `hr` | **100 %** | 52,718 | **885** | 744 |
| `temp` | 8.0 % | 3,867 | 68 | 62 |
| `hrv` | 6.1 % | 2,828 | 49 | 55 |
| `steps` | 2.8 % | — | — | — |
| `spo2` | **0.001 %** (21 readings in 2.17M) | — | — | — |

Three consequences:

- **HR gives 18.6x the readings and 18x the readings per unbroken segment.** That is
  what decides foundation-model feasibility: MOMENT and Mantis both want 512
  *contiguous* points, which a 49-reading HRV segment cannot fill and an 885-reading HR
  segment can.
- **`spo2` is unusable.** 21 readings in 2.17M rows. It is in `CHANNELS` so the loader
  does not special-case it, and `channels()` drops it for every patient.
- The 30 patients with heart rate but no surviving HRV now get an `hr` file. `ingest`
  still reports them as "failed", because that status has always meant the HRV verdict
  and the documented cohort counts keep their meaning; the log's `channels` column says
  what was actually written.

**`--signal hrv` is byte-identical to the single-channel pipeline** — same filename, same
columns, same artifact names and keys (the `signal` param enters the key only when it is
not the default). Verified by re-deriving 10 patients' HRV files and comparing hashes.

## Six pools, because four operations have four output shapes

| pool | entry point | emits | why separate |
|---|---|---|---|
| `smoothing/` | `smooth_dataframe` | a value per reading | estimates the level AT t |
| `forecasting/` | `forecast` | a value **+ 21 quantiles** per reading | predicts t+h; its residual is SURPRISE, not noise |
| `cpd/` | `detect` | a flag per reading | — |
| `encoders/` | `encode` | a **vector** per patient-day | a feature block, not a prediction |
| `models/` | `fit`/`predict_proba` | a probability per day | — |
| `metrics/` | `compute` | a number | — |

A forecaster is **not** a smoother. Putting it in `smoothing/` would make
`sweep smoothing` compare two different quantities (rule 5 says a comparison varies only
the method), and the smoothing contract has nowhere to put a predictive distribution —
which is the part worth having. `tail_<sig>` is the observed value's position in that
distribution, in [0, 1]: "this reading fell at the 99.5th percentile of what the model
expected" is calibrated and comparable across patients and channels, while "6 ms off"
means different things at a 30 ms and a 90 ms baseline.

`detect --on` therefore takes **five** signals now: `raw`, `smoothed`, `residual` (from
a smoother) and `surprise`, `tail` (from a forecaster).

### The `encode` step and why it is not part of `model`

A frozen forward pass does not depend on the fold, the seed, the head or the task. Run
inside a model it is recomputed per (model x fold x seed) and its output can reach
neither `cpd/` nor the other task. Run as a step it is computed once, content-keyed, and
reused by every fold, seed, model and both tasks — the only reason a 0.3B encoder is
affordable here.

- **`grain: "day"`** in the manifest: an encode artifact is one row per patient-day, so
  `join` passes its columns straight through. Without the flag a 1,024-wide embedding
  becomes 4,096 columns (three quarters a std/min/max over a single row) and
  `_day_features` raises, because those rows have no `createdTime`.
- **No reduction in the step, deliberately.** PCA belongs inside the model, where the
  train/test split exists. Reducing in `encode` would fit on test patients — it runs
  before the split is drawn.

## The context budget: `days x profile` IS the input length

`wearable/windowing.py` is the one implementation of the window rule (the `chunking.py`
precedent), shared by `join` and `encode`. Never redefine it; import it.

Foundation models have a fixed ceiling, so the window is not a free choice:

| model | ceiling |
|---|---|
| MOMENT | **exactly 512** (its positional embeddings are pretrained there) |
| MantisV2 | a multiple of 32, pretrained at 512 |
| Timestral | up to 32,768 (1,024 patches of 32) |

**Changing `--days` against a fixed ceiling trades history length for time resolution; it
does not change how much the model is handed.** 7 days into 512 slots is ~20 min a slot,
1 day is ~2.8 min. That is a different experiment from the `sweep window` ablation,
where more days really do mean more information.

And it interacts with the channel:

- HRV at ~55 readings/day: 7 days = **385 points, fits 512 natively**, nothing invented.
- HR at ~744 readings/day: 7 days = **5,208 points, 10x over budget** — use
  `windowing.bins_for(7, 512) == 73`, i.e. `--profile 73` (511 points).

`encoders/_resample.py` handles the mismatch when it cannot be avoided, and the mode is
keyed so the choices are comparable: `pad` (right-aligned, masked, nothing invented —
needs the window to fit and a model that takes a mask), `interpolate` (Mantis's own
recommendation; **invents values across gaps and rescales time per row**, so a 20-reading
and a 140-reading day both become 512 points), `decimate` (every k-th slot; loses detail
but never invents a value).

### Frozen, therefore causal

The forecasters are **truncation-exact by construction**: weights frozen, nothing fitted
on the cohort, each block predicted from strictly earlier readings. Of the ten smoothers
only `anrewma` is. That makes forecast surprise the one residual in this repo with no
lookahead to argue about — see the H9 section. `forecasting/_base.py` has a self-check
that proves it with a stub and no weights (`python -m wearable.forecasting._base`).

### New input variants

`forecast@<f>` · `surprise+cpd@<f>` · `embed@<e>`, beside the original five, **generated
from the pools**, so each forecaster and encoder has its own variant. Adding a model file
adds its variant, and the model name sits in the `input` column of the runs table, so two
forecasters never overwrite each other's row. A bare family name expands:
`--variants surprise+cpd` runs all three forecasters. A variant is four switches
(`smooth`, `detect_on`, `forecast`, `encode`) and nothing else.

**Channels.** A path's recipe runs once per channel in `--signals` (default
`config.PATH_SIGNALS = ("hrv", "hr")`), each on its own readings and as its own feature
block. Every other channel still rides along raw. `signals` is a column of the runs table
and part of a cell's identity. Detect and encode artifacts on a non-HRV channel are
named `<method>@<signal>`, as smooth and forecast already were. Without that, HR and
HRV would share a column prefix in `join`. Cost: HR has ~16x HRV's readings, so the first
HR detect pass is the slow one (cached after).

## Change-point detection: noise model, calibration, causality

Design and the measurements behind it: https://claude.ai/artifact/CDgdpCNWBYfiZrqGdNiVh9
(decisions of 2 Nov 2026). Five detectors (`bocpd kcpd kalman hmm drift`, all in
`DEFAULT_ENSEMBLE`; `drift` compares each night's *sleep* level, 18:00–12:00, with the
patient's recent nights) read the column through a per-patient, per-channel **noise model**,
`cpd/_whiten.py` (a `_` helper, not a method):

- **Binning.** HR is median-binned to 5 min first (`bin_min`). Its correlation time is
  ~9 min, so little is lost, and 1–5-min surges (posture, movement) stop dominating.
  Each reading holds the output of the latest *closed* bin (causal).
- **Arming.** Nothing alarms before the patient's own first worn day (`burn_in_min` =
  1440 worn minutes). There is no population prior; that was a decision.
- **Whitening.** The residual is centred and scaled by hour of day (smoothed ±2 h), then
  corrected by behavioural state. That state is SLEEP (sleep_active), ACTIVE (awake,
  steps_active) or REST (awake, no steps), taken over the minutes a reading covers (HRV
  10, HR 1). Each state gets its own centre offset and scale ratio, shrunk toward none
  when rare; no flags means UNKNOWN, with no correction. Measured, clock hour explains
  the spread about 2× better than state, but state adds to it, so it is hour × state.
  Then a Huber-robust Kalman filter on an OU + white-noise model turns it into
  innovations `z`. A long gap decays its memory, and an H8 reset clears it. It is refitted at arming
  and every midnight from the trailing `baseline_days`, skipping abnormal days
  (>`abnormal_frac` of |z|>3).
- **Settle windows** after a re-entry (`reentry_gap_min`) or an H8 reset: detectors
  update but stay unarmed. The level jump across the gap goes to the step-level columns
  `reentry_jump_z` / `reentry_gap_min`, never to an alarm.
- **Calibration ("jumpiness").** No labels are used. A no-change *twin* is built from
  whole days of the patient's own trailing data, resampled in shuffled order and run
  through the same noise model. Each detector's `score()` is thresholded on the twin so
  that the ensemble fires `budget` times per worn day, split evenly across detectors
  and their statistics. It is rebuilt every `recal_days`, so thresholds are
  per-reading arrays. `drift` uses an analytic CUSUM threshold (`threshold_for`). The
  budget is a detect param keyed in the cache, `config.CPD_BUDGETS = (1/7, 1/3, 1)`, and
  a sweep axis: `--cpd_budget 0.142857,0.333333,1.0`. `cpd_budget` is a runs column.
- **Calibration method** (`config.CPD_CALIBRATION`, `detect --calibration`,
  `sweep ... --cpd_calibration shuffle,sts`). `shuffle` (default) is the day-resampled
  twin above. `sts` (`cpd/_sts.py`) fits a per-patient structural time-series model on
  past readings at native resolution: log scale, fixed level, a deterministic 24 h
  harmonic seasonal, AR(2), sleep/active regressors and Student-t noise, on a clock-time
  grid with the gaps as NaN. It simulates new values on the history's own observation
  mask, in independent weekly blocks, and pushes them through the *same* smoother,
  binning, noise model and detectors. A random-walk level was tried and rejected: it
  wanders unboundedly (simulated HR 50–190 bpm against a real 80–115). Under 3 days of
  history the circadian term is dropped; a failed fit falls back to shuffle for that
  recalibration, recorded as such. `--on surprise|tail` raises: it would need the
  forecaster run per simulation. `STS_SETTINGS` are keyed only for `sts`, so the
  default's keys are untouched. Both methods write a `calibration` table beside
  `detected` (per patient × recalibration: method used, STS parameters, every threshold,
  the achieved no-change rate), and `scripts/cpd_calibration_compare.py` compares them.
  Known limits: AR(2) fits mostly settle on one root, so hours-scale autocorrelation is
  under-modelled (HR at 60 min: 0.43 synthetic against 0.62 real), and STS cannot
  reproduce the HRV pile-up at 120–129.
- **Settings** are in minutes, per channel, in `config.CPD_SETTINGS`. The `detect`
  block holds the noise model's; the others override each detector's `DEFAULTS`. A
  misspelt name raises. Everything is in the detect cache key. `wearable list` prints
  both channels.

**Every piece is causal.** `selftest` runs the whole detect step on synthetic patients,
on a prefix and on the full series, and asserts they agree. The HMM filters forward;
there is no Viterbi and no EM. Two things are *not* exactly as designed:
- **Measured alarm rates are about 4× the budget on real data.** Full cohort at 1/3:
  HRV 1.39 and HR 1.45 events per worn day (per-patient median 1.45 / 1.61), with
  8.6% / 9.8% of armed time flagged. The original detectors gave 0.8 and 37.7 events
  and flagged 48% of HR. Real data contains real changes, and the twin cannot show
  everything. Read the budget as a sensitivity *setting* that orders the three detect
  builds, not as a guaranteed alarm rate. `drift` flags only the reading where its
  nightly verdict arrives, and restarts its CUSUM after each signal.
- **The first week runs hotter.** Thresholds there come from a twin of a single worn
  day, which understates how much later days differ.

**The encoder window follows `--days`.** `experiment.inputs` embeds the `days` window
ending on each day at `profile = bins_for(days, CONTEXT)`, so the ablation varies
resolution, not budget. `join` does **not** lag an embedding (it already covers the
window), and an embedding never displaces its channel's raw block (`join4`).

## A foundation model AS the classifier — `models/mantis_clf.py`

Every other foundation route uses a pretrained model in the middle (surprise, embedding)
and a small model predicts. `mantis_clf` is the other arrangement: MantisV2 reads the
recorded series and emits the probability itself.

**Why Mantis and not MOMENT.** 144 positive days. MantisV2 is **8.1M** parameters;
MOMENT-base is ~110M and large ~385M. Freezing a 385M body and training a linear head is
not "using the model directly" — it is `encoders/moment.py` with a weaker head, since
`join` plus a tree model can put anything on top of an embedding. Mantis is the only one
small enough that end-to-end training here is a real option. It also takes several
channels at once and can learn a combination of them
(`LinearChannelCombiner`, trained with the head), which frozen per-channel embeddings
cannot — each is embedded in ignorance of the others.

**Input: the recorded series, so it needs profile columns.**
`windowing.channel_series` groups them by channel and orders each oldest-reading-first.
That is a *different reading of the same columns* from `models/_seq.sequence`, which
treats the DAY as the time step for `rnn`/`lstm`. Here the time step is the reading.

```bash
pip install mantis-tsfm
python -m wearable ingest                       # all channels, needed for multivariate
python -m wearable join --channels hrv,hr,temp --days 7 --profile 55
python -m wearable sweep inputs --variants raw --days 7 --profile 55 \
    --channels hrv,hr,temp --model_names mantis_clf,xgboost,always_no
```

There is no new `INPUTS` variant: this is the `raw` route with `--profile` set, because
what changes is the table join builds, not which artifacts it receives.

`fine_tuning_type` is the knob that matters — `full` trains the whole body (the actual
experiment) and `head` freezes it (the conservative reading, and all a 385M model could
do). Both are keyed, so they are comparable.

**Expect it to lose.** The reference design ran the equivalent: TimesFM 3 frozen with a
learned logistic layer scored onset AUROC 0.552 [0.44-0.66] and detection **0.499** —
chance — while its purpose-built end-to-end network managed 0.506 and was beaten by its
own ablation. Trees were among the better rows. Pretraining transfers knowledge of what
time series look like, not of what an exacerbation looks like; that mapping comes from
the 144. A clean negative is worth reporting, because readers will ask.

**Cost.** Measured at ~0.8 s per epoch per 60 samples on this CPU, so roughly 170 s an
epoch over 12,889 training days — about an hour per fold per task at 20 epochs on CPU,
far less on a GPU.

## The evaluation design: three populations, one principle

**Keep the population fixed, change only the method, and measure whether prediction
actually improves.** That is what makes a claim about a smoother, a detector or a
foundation model credible, and it is why `join`'s universe comes from the LABELS and not
from whichever method produced features — a patient a method dropped yields NaN
features, never a missing row, so every arm is scored on identical test patients.

| tier | set | `--cohort` | patients | days | exac days | base rate | role |
|---|---|---|---|---|---|---|---|
| 1 | every labelled patient | `all` (default) | **191** | 15,691 | 144 | 0.92 % | **primary** |
| 2 | smoothable (>=`MIN_SEGMENT_ROWS` in a chunk) | `smoothable` | 187 | 15,685 | 144 | 0.92 % | robustness |
| — | CPD-armable (>=`BURN_IN_N`) | `armable` | 180 | 15,662 | 144 | 0.92 % | robustness |
| 3 | has any exacerbation day | `exacerbators` | 38 | 4,976 | 144 | **2.89 %** | secondary, labelled |

`join.analysis_set(name)` resolves each from the labels and the ingested series — never
from an artifact, so a set does not depend on which method was run. Restriction is
**keyed**: a 187-patient result can never be served from a 191-patient cache.

Three things measured, not assumed:

- **All ten smoothers fail on exactly the same 4 patients** (1, 3, 5 and 10 readings).
  Eligibility is set by the shared `MIN_SEGMENT_ROWS`, not by any method's character, so
  the smoother comparison is not threatened by it.
- **Every patient any data-quality threshold excludes is a non-exacerbator** — 0 of the
  4, 0 of the 11 unarmable. Restricting nudges the exacerbator share *up* (19.9 % ->
  20.3 %), the wrong direction. Report it rather than let a reviewer find it.
- **The stratified split is not stable under restriction.** Dropping those 4 moved one
  patient across the boundary in 3 of 5 seeds, and a patient carries a median of 71
  sample days against the 4 patients' 6 in total — so a naive tier-2 run measures a
  6-day effect through a 71-day confound. For tier 2 either state it arithmetically (6
  of 15,691 days, 0 of 144 positives) or run it with `stratify=False`, where the split
  is a pure function of (seed, id) and nothing moves. Tier 3 *should* re-draw the split:
  it is a genuinely different population.

**Tier 3 selects on the outcome**, which is why it is never the headline: it inflates
the base rate 3.1x, discards 10,715 negative days, and makes
`false_alarms_per_30_days` unmeasurable in the 80 % of patients you would deploy to. At
deployment you do not know who is about to exacerbate — that is the prediction problem.
It answers a different and legitimate question: *among patients who do exacerbate, can
we separate their bad days from their good ones?*

```bash
python -m wearable sweep inputs --task detection                      # tier 1
python -m wearable sweep inputs --cohort smoothable --stratify False   # tier 2
python -m wearable sweep inputs --cohort exacerbators                  # tier 3
```



**All 191, not the 38 with an exacerbation.** Selecting patients on the outcome is
selection on the label:

| | patients | sample days | exac days | base rate |
|---|---|---|---|---|
| all | 191 | 15,691 | 144 | **0.92 %** |
| exacerbators only | 38 | 4,976 | 144 | **2.89 %** |

Restricting throws away 10,715 negative days and 153 of 191 patients, inflates the base
rate 3.1x, and leaves the model never having seen what a never-exacerbating patient's
normal looks like — which is 80 % of the people it would be used on. It also makes
`false_alarms_per_30_days` unmeasurable in that population, and at deployment you do
not know who is about to exacerbate, which is the whole point.

The split already handles this correctly: `_strata` stratifies by cohort group x
has-any-exacerbation, so both kinds of patient land in train and test in proportion.
Class imbalance is the model's job (`class_weight="balanced"`, `scale_pos_weight`), not
the cohort's.

A restriction to exacerbators is legitimate only as a clearly-labelled **secondary**
analysis ("among patients who do exacerbate, can we tell their bad days from their good
ones?"). Restricting on *data volume* is a different and defensible thing — the 11
patients with <144 valid samples never arm CPD — as long as it is applied identically to
train and test.

## Other invariants `selftest` pins

- H8: a gap ≥180 min splits, <180 does not, ids restart per patient. **ids start at 1**
  — the first row's NaN dt counts as a boundary.
- **A deliberate inconsistency:** ingest fills gaps on `> 180` while the chunker splits
  on `>= 180`, so a gap of exactly 180.000 min opens a new chunk but gets no filler
  row. Preserved from the original Stage 0 because changing it moves every downstream
  output; the 10-min cadence makes it vanishingly rare.
- The split is stable when the patient universe grows (hashing ids with the seed, not
  shuffling), so adding a cohort cannot move existing patients between train and test.
- Every model metric ranks a good fit above a bad one; signal metrics are finite on
  real input and NaN on too little data.

## Data

| Source | Input | Cohort | Labels |
|---|---|---|---|
| `pass1` (default) | `data/raw/pass1/<cohort>/<pid>.csv.gz` | 221 `in_models` (AIIMS a/p + COPD) | **yes** |
| `spark` | `data/raw/spark/<pid>/hrv/*.csv` | 109, old ID scheme | no |

`pass1_data`'s `hrv` column *is* the old `hrvValue`: watch SDNN over ~10 min, one value
~every 10 min, device-capped at 129. 191 of 221 patients yield data (30 have heart rate
but no surviving HRV); 11 of those have <144 valid samples so CPD never arms them.

**Ground truth — `data/processed/labels_pass1.csv`** (`config.LABELS_CSV`): one row per patient-day, collapsed off the
minute grid *before* HRV filtering, so an exacerbation day carrying no HRV is still on
record. Counts reproduce `data/raw/pass1/README.md` §5 exactly: 15,691 sample days, 144
exacerbation days, 151 onset days, 100 episodes. `data/raw/pass1/README.md` is
authoritative for that cohort. The file also carries `group` (aiims_adult / aiims_paed /
copd) from `patients.csv`, which the minute files do not have: the modelling split is
stratified by it and scores are reported per group, so **a labels file written before
that change must be re-ingested**.

**`data/` is machine-local and may be absent entirely** (a fresh clone has none).
`ingest` needs `data/raw/pass1/patients.csv` plus the per-cohort files; check with
`ls data/` before assuming any state. After the last reset, `data/raw/pass1/` was the
only input and everything earlier went to `data/old/` (see `data/old/README.md`), which
nothing reads — except the pf row of `scripts/smoother_comparison_report.py`, via
`config.smoothed_dir`. That script defaults to patient `0010`, a spark patient; pass a
pass1 id. The `spark` source is still registered but its export moved to
`data/old/raw_spark/`, so it reports 0 patients.

Where `data/old/` exists, two things in it are **not reproducible** by the current
code: the stage-3a adaptive-baseline output (`calibrated_`/`annotated_gammadglm`, since
3a is unwired in `legacy/`) and the non-causal pf output (`pf_smoothed/`, since pf is
not in the smoothing registry). Copy them back before deleting `data/old/`.

**63 patient IDs collide between the two sources** (`a001`, `p017`, …) — never mix
spark and pass1 output in one `data/processed/`.

**HRV data is sensitive — never commit anything under `data/` — raw cohorts, the artifact store, or any
patient CSVs to a public remote.** The one exception is an allowlist in `.gitignore`: seven
cohort-level files in `data/reports/smoothing/` (`cohort_stats.json`, `ideal_reference.json`,
`methods.html`, `metrics_summary.csv`, `runs.csv`, `settings.csv`, `settings.json`), which carry
no patient ids or readings. The dashboard, `patient_stats.json` and `metrics_per_patient.csv`
are per-patient and must stay out. Before you add anything to that allowlist, check that it is
aggregate-only.

## Conventions

- Never redefine `GAP_THRESHOLD_MIN` or `assign_chunks`; import them. Tests enforce it.
- Never import one step from another. Compose in `experiment.py`. The one allowed
  exception, pinned by the test: `join` imports `detect._load_input` for the raw-only
  baseline. `scripts/` may import anything (`plot_patient.py` uses `__main__._resolve`).
- Prefer data-driven per-patient priors over hard-coded scales (H10).
- Determinism: `kcpd` seeds `np.random.default_rng(0)`, each calibration twin is seeded
  by its position, and models take an explicit seed. Given identical inputs, outputs are byte-reproducible —
  the restructure to the package was verified this way (192/192 processed, smoothed and
  calibrated files byte-identical).
- Parallelism: ingest, smooth and detect use `ProcessPoolExecutor` (`cpu_count − 1`;
  detect parallelises over patients). Calibration dominates detect: the twin is rebuilt
  weekly and every detector is scored on 60 worn days of it, so an HR patient takes
  ~1.5 min of one core.
