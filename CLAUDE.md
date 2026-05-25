# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

This repo is **scaffolding only**. Every pipeline `.py` file under `pipeline/` is an
empty 0-byte skeleton — there is no implementation yet. `README.md` describes the
intended design; the files exist so stages can be filled in one at a time. Do not
assume any stage runs until you have implemented it. There is no test suite, linter,
or build step configured.

The git repo root is the `asthma-prediction/` directory (it has its own `.git`).
A fresh shell starts one level up in `asthma-predict/`, so `cd` into
`asthma-prediction/` before running anything.

## What this pipeline is

A four-stage pipeline that aligns multi-signal wearable data onto a shared
integer-minute axis, then smooths and runs change-point detection (CPD) on **HRV
only**. Each stage reads the previous stage's CSV and writes the next:

```
data/raw_data/<patient>/   ── Stage 1 preprocessing ─▶ data/processed/<patient>_aligned.csv
                           ── Stage 2 cleaning       ─▶ data/processed/<patient>_cleaned.csv
                           ── Stage 3 smoothing      ─▶ data/processed/<patient>_smoothed.csv
                           ── Stage 4 cpd_pipeline   ─▶ pipeline/outputs/<patient>_final.csv
```

`pipeline/run_pipeline.py <patient_id>` is meant to orchestrate all four in order.
`pipeline/config.py` is the single source of paths (derived from
`Path(__file__).parents`, never hardcoded) and constants.

## Non-negotiable design invariants

These are the rules that make the pipeline correct; preserve them in every stage:

- **HRV is primary.** HRV readings define the base grid (one row per HRV reading on an
  integer-minute axis from the common start). Smoothing and CPD operate on **HRV only**.
- **Carry-through.** Every other signal (HR, Steps, Temp, SpO2, Sleep) is aligned once in
  Stage 1 and carried *unchanged* through all four stages. The CPD stage especially must
  not drop context columns when it assembles the final frame.
- **Common start = the latest first-reading across signals**, so all sensors are active
  before row 0. Data before it is discarded.
- **Gaps stay gaps.** Stage 2 splits the HRV series at large time gaps (`minute_diff` >
  file mean gap) by masking those values to NaN, and never imputes the removed data.
- **Detectors stay separate.** Stage 4 runs four detectors (bocpd, kcp, hmm, kalman) and
  keeps each as its own `*_indicator` / `*_magnitude` columns — never merged or averaged.
- **No composite/derived score columns.** Only raw aligned values plus detector outputs.
- **Patient-general.** A new patient is a new `data/raw_data/<id>/` folder and nothing
  else; no hardcoded dates, row counts, or patient ids.

## Raw data layout (real, differs from README casing)

`data/raw_data/<patient_id>/` holds six **lowercase** signal subfolders, each a Spark
export with a `part-*.csv` plus `_SUCCESS` and `.crc` sidecars that must be ignored
everywhere. README documents them as `HRV/`, `HR/`, `Temp/`, etc.; on disk they are:

| Folder         | Columns used                                | Role in pipeline                |
|----------------|---------------------------------------------|---------------------------------|
| `hrv`          | `createdTime`, `hrvValue`                    | **primary** (base grid)         |
| `heartrate`    | `logDateTime`, `lastRate`                    | aggregate denser readings → bin |
| `temperature`  | `createdTime`, `temperature`                 | carry within tolerance          |
| `steps`        | `logDateTime`, `logEndTime`, `steps`, `distance` | aggregate (sum)            |
| `spo2`         | `createdTime`, `spo2Value`                   | sparse — NaN where no near reading |
| `sleep`        | `logDateTime`, `logEndTime` (+ ignored cols) | → single binary `is_asleep`     |

Gotchas confirmed in the data:
- **Match folder names case-insensitively** (the README spec uses different casing).
- Rows are **not time-sorted**; sort after parsing. Timestamps are ISO strings, some with
  a `.000` fraction (`heartrate`).
- **Missing signals are normal**: ~7 patients have no `spo2`, 2 have no `sleep`. Handle a
  missing folder as an all-NaN / all-zero carried column, not an error.
- Some patients have an extra `realtimetemphr/` folder — ignore unknown folders.
- Patient ids are heterogeneous: `a001`, `p018`, numeric like `0010`/`005`. Don't assume a
  prefix or format.
- `data/raw_data/` is **gitignored** (clinical data — 109 patients, ~310 MB) except its
  `.gitkeep`.

## Final output columns (Stage 4)

```
timestep, real_value, smoothed_value,
bocpd_indicator, bocpd_magnitude, kcp_indicator, kcp_magnitude,
hmm_indicator, hmm_magnitude, kalman_indicator, kalman_magnitude,
HR, steps, temperature, spo2, is_asleep,
<signal>_mask..., watch_off_flag, clip_flag, timestamp
```

Indicators are 0/1; magnitudes are in [0, 1]. `watch_off_flag` = temperature below
threshold; `clip_flag` = HRV outside a plausible physiological range (the README spec
lists `clip_flag` but does not define it — it is a flag, not a value transform).

## Commands

Run from the `asthma-prediction/` repo root so package imports (`from pipeline import
config`) resolve.

```bash
pip install -r requirements.txt

# full pipeline for one patient (once stages are implemented)
python pipeline/run_pipeline.py <patient_id>

# train the VAE used by Stage 3 (writes pipeline/checkpoints/vae_cnn.pt)
python -m pipeline.smoothing.train_vae <patient_id>

# run a single stage in isolation while developing
python -m pipeline.preprocessing.preprocessing <patient_id>
```

Stack: pandas/numpy/scipy for alignment & cleaning, torch for the CNN-VAE smoother,
`ruptures` (kcp), `hmmlearn` (hmm), and a hand-rolled BOCPD / Kalman for Stage 4.
