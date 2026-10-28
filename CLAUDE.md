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
  smoothing/       10 methods        smooth_dataframe(...)
  cpd/             4 methods         detect(time, values, priors)
  models/          1 model           fit(...) + predict_proba(...)
  metrics/         9 metrics         compute(...) + KIND
  steps/           ingest · smooth · detect · join · model · evaluate
  experiment.py    the ONLY thing that connects steps: chain() and the sweeps
  selftest.py      asserts the five rules
  __main__.py      the CLI
scripts/           reporting: eda, plot_patient, smoother_comparison_report
legacy/            retired code (see legacy/README.md)
docs/              method notes, the PDF and DOCX
tests/             smoke tests — enforce rules 1, 2, 4, 5
data/              see the naming rule below; never committed
  raw/{pass1,spark}/   inputs, never written by the pipeline
  processed/           ingest output + labels_pass1.csv
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

python -m wearable ingest [--source pass1|spark]
python -m wearable smooth gammadglm [--deltrend 0.995] [--force]
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
python -m wearable sweep smoothing --limit 40 # first 40 patients; limit is in the key
```

`smooth` and `sweep` take `--<knob> <value>` pairs parsed as Python literals and passed
straight through as kwargs (so `--limit` works there; `chain` has no `--limit`).

**Artifact specs** are `step:method[:key]` with the split at the *first* colon (method
names contain colons: `join:smooth:gammadglm`, `evaluate:model:logreg`). Without a key
the newest complete artifact of that name wins. Detect artifacts are named by the
ensemble (`detect:bocpd+kcpd+kalman+hmm`), not by `--on`/`--smoothing`, so runs on
different signals share a name — pass the key from `runs` when that matters.

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

## What the cache key does not see

`key = sha256(step, method, params, input keys, code_hash)`, where `code_hash` is the
**method module's file hash** only. So a reuse can be stale when you change:

- **step code** (`wearable/steps/*.py`), `chunking.py`, or constants in `config.py`
  (`GAP_THRESHOLD_MIN`, `MIN_SEGMENT_ROWS`, …) — none are hashed. Every smoother imports
  `chunking`/`config`, yet editing them invalidates nothing.
- **join logic** — join's `code_hash` is the literal `"join1"`; bump it in
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
| a smoother | `wearable/smoothing/<name>.py` with `smooth_dataframe` | nothing — it is discovered |
| a detector | `wearable/cpd/<name>.py` with `detect` | add the key to `DEFAULT_ENSEMBLE` to put it in the vote |
| a model | `wearable/models/<name>.py` with `fit` + `predict_proba` | nothing |
| a metric | `wearable/metrics/<name>.py` with `compute` + `KIND` | nothing — it appears in every comparison |

Optional module attributes: `LABEL`, `DEFAULTS` (the knobs — these replaced the old
`GLOBAL_*` constants), `CONFIG` (the dataclass), `KIND` (metrics: `"model"` takes
`y_true, y_prob`; `"signal"` takes `raw, smoothed`), `HIGHER_IS_BETTER`.

A module whose name starts with `_`, or which lacks the entry point, is skipped — so
helpers can live in a pool directory. A method that fails to import is reported as
unavailable rather than taking the registry down — but `selftest` fails on any
unavailable method, so a new dependency must go in `requirements.txt`.

A new smoother must emit a strictly positive baseline (H1, asserted). If it is
truncation-exact (H9, below), add it to `TRUNCATION_EXACT` in `wearable/selftest.py`;
selftest then fails if it ever regresses.

## Signal choice: raw, smoothed, or residual

`detect --on` picks the column. `residual_hrv = hrvValue − smoothed_hrv` is computed by
the smooth step because it is a property of smoothing, **not** the architecture it used
to be. The research rationale still holds — CPD on raw false-alarms on circadian drift,
CPD on smoothed erases the volatility bursts that signal an attack — but it is now one
option of three and `sweep signal` measures whether it actually wins.

## H-Framework

H1 positivity, H2/H4 drifting multimodal baseline, H5 circadian, H7 irregular sampling,
**H8 ≥180-min segmentation (state flush)**, **H9 strict causality**, H10 per-patient
adaptation.

**H9 is weaker than it looks — do not claim it repo-wide in the paper.** `selftest`
probes it by truncation (does a prefix run reproduce the full run where they overlap?)
and finds **only `anrewma` and `ctimm` are truncation-exact**. The other eight filter
forward but fit their per-patient parameters (H10: MAD-noise, log-mean, observed range,
median dt) on the *whole frame handed in*, so a value at time t moves when later rows
arrive. That is a lookahead path, and a residual computed with parameters that saw the
future is an optimistically biased early-warning signal. Measured deviations:
`logmhw 0.479, rspf 0.388, gpssm 0.172, krlst 0.135, kim 0.122, ossa 0.111,
gammadglm 0.052, hrkf 0.002`. This is pre-existing, not a regression — identical before
and after the restructure. To close it, derive those parameters from a causal expanding
window.

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
authoritative for that cohort.

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
patient CSVs to a public remote.**

## Conventions

- Never redefine `GAP_THRESHOLD_MIN` or `assign_chunks`; import them. Tests enforce it.
- Never import one step from another. Compose in `experiment.py`. The one allowed
  exception, pinned by the test: `join` imports `detect._load_input` for the raw-only
  baseline. `scripts/` may import anything (`plot_patient.py` uses `__main__._resolve`).
- Prefer data-driven per-patient priors over hard-coded scales (H10).
- Determinism: `kcpd` seeds `np.random.default_rng(0)`, `hmm` uses `random_state=42`,
  models take an explicit seed. Given identical inputs, outputs are byte-reproducible —
  the restructure to the package was verified this way (192/192 processed, smoothed and
  calibrated files byte-identical).
- Parallelism: ingest and smooth use `ProcessPoolExecutor` (`cpu_count − 1`); detect
  runs patient-by-patient. KCPD's permutation test is the heaviest step — a full
  4-detector detect is ~10-15 min for 188 patients.
