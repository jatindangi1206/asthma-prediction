# HRV Change-Point Pipeline

A four-stage pipeline that aligns multi-signal wearable data onto a shared
minute axis, then smooths and runs change-point detection on **Heart Rate
Variability (HRV) only**. Every other signal (HR, Temp, Steps, SpO2, Sleep) is
aligned and **carried through untouched** — never smoothed, never detected on.

## Pipeline stages

```
raw_data/<patient>/        six signal subfolders (Spark part-*.csv)
        │
        ▼  Stage 1  preprocessing/preprocessing.py
data/processed/<patient>_aligned.csv      one row per minute, all signals aligned
        │
        ▼  Stage 2  cleaning/cleaning.py
data/processed/<patient>_cleaned.csv      HRV gap-chunked (gaps stay gaps)
        │
        ▼  Stage 3  smoothing/smoothing.py   (uses pipeline/checkpoints/vae_cnn.pt)
data/processed/<patient>_smoothed.csv     + smoothed_value (HRV)
        │
        ▼  Stage 4  cpd/cpd_pipeline.py
pipeline/outputs/<patient>_final.csv      4 detectors as separate columns + context
```

## Data layout

`data/raw_data/<patient_id>/` contains six subfolders, each with a Spark-style
`part-*.csv`. `_SUCCESS` and `.crc` sidecars are ignored everywhere.

| Subfolder | Columns used                                   | Role |
|-----------|------------------------------------------------|------|
| `HRV/`    | `createdTime`, `hrvValue`                       | **primary** signal |
| `HR/`     | `logDateTime`, `lastRate`                       | context (aggregated) |
| `Temp/`   | `createdTime`, `temperature`                    | context (carried) |
| `Steps/`  | `logDateTime`, `logEndTime`, `steps`, `distance`| context (aggregated) |
| `SpO2/`   | `createdTime`, `spo2Value`                      | context (sparse, NaN gaps) |
| `Sleep/`  | `logDateTime`, `logEndTime` (others ignored)    | context (-> `is_asleep` binary) |

## Usage

```bash
pip install -r requirements.txt

# (optional) train the VAE on a patient's cleaned HRV first
python -m pipeline.smoothing.train_vae <patient_id>

# run all four stages in order
python pipeline/run_pipeline.py <patient_id>
```

Each stage reads the previous stage's output, so a new patient is just a new
folder under `data/raw_data/` — no code changes, no hardcoded dates or row
counts.

## Final output columns

```
timestep, real_value, smoothed_value,
bocpd_indicator, bocpd_magnitude, kcp_indicator, kcp_magnitude,
hmm_indicator, hmm_magnitude, kalman_indicator, kalman_magnitude,
HR, steps, temperature, spo2, is_asleep,
<signal>_mask..., watch_off_flag, clip_flag, timestamp
```

Detector outputs are kept as **separate columns** and never merged or averaged.
There are no composite/derived score columns — only raw aligned values plus
detector outputs.

## Design rules

- **HRV-only processing.** Smoothing and CPD touch HRV exclusively; all other
  signals are carried through every stage unchanged.
- **Gaps stay gaps.** Cleaning splits the HRV series at large time gaps and does
  not impute the removed data.
- **Generalises across patients.** No hardcoded dates or row counts; sidecar
  files are ignored; one code path for all patients.
