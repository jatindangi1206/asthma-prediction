"""Stage 2 — cleaning (HRV only; context carried through untouched).

Reads ``<id>_aligned.csv``, operates on HRV + the minute axis only, and writes
``<id>_cleaned.csv``. Steps (per the agreed design):

1. Mean method for HRV missingness — fill NaN HRV with the series mean.
2. Add ``minute_diff`` (diff of the minute axis) and the file-level ``mean`` gap.
3. Chunk: where ``minute_diff > mean`` gap, mask the HRV value to NaN so the
   series splits at large time gaps.
4. Do NOT impute the chunked-out data — gaps stay gaps.

Raw HRV (``hrvValue``) is preserved untouched (it becomes ``real_value`` in the
final CSV); cleaning writes a separate ``hrv_clean`` column for downstream
smoothing. Every other column rides through unchanged.

This supersedes the partial helper ``preprocessing/add_time_stats.py`` (which
only added minute_diff + mean); that file is left in place but unused.

    python -m pipeline.cleaning.cleaning <patient_id>
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline import config


def clean_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, int, float]:
    """Apply Stage-2 cleaning to an aligned frame.

    Returns (cleaned_frame, n_gap_splits, gap_threshold).
    """
    out = df.copy()
    hrv = pd.to_numeric(out[config.HRV_RAW_COL], errors="coerce")

    # 1. mean-fill HRV missingness
    hrv_clean = hrv.fillna(hrv.mean())

    # 2. minute_diff + file-level mean gap
    minute = pd.to_numeric(out[config.MINUTE_COL], errors="coerce")
    minute_diff = minute.diff().fillna(0.0)
    mean_gap = float(minute_diff.mean())
    out[config.MINUTE_DIFF_COL] = minute_diff
    out[config.MEAN_GAP_COL] = mean_gap

    # 3. chunk: mask HRV to NaN where the gap exceeds the mean gap (split series)
    threshold = config.GAP_MULTIPLIER * mean_gap
    gap_mask = minute_diff > threshold
    hrv_clean = hrv_clean.mask(gap_mask)   # 4. NOT imputed -> stays NaN

    out[config.HRV_CLEAN_COL] = hrv_clean
    return out, int(gap_mask.sum()), threshold


def clean_patient(patient_id: str) -> Path:
    config.ensure_dirs()
    src = config.processed_path(patient_id, "aligned")
    if not src.exists():
        raise FileNotFoundError(f"Run Stage 1 first; missing {src}")
    print(f"=== Stage 2: cleaning {patient_id} ===")
    df = pd.read_csv(src)
    out, n_chunks, threshold = clean_frame(df)
    dest = config.processed_path(patient_id, "cleaned")
    out.to_csv(dest, index=False)
    n_nan = int(out[config.HRV_CLEAN_COL].isna().sum())
    print(f"[{patient_id}] mean gap={threshold:.2f} min, "
          f"{n_chunks} gap splits -> {n_nan} HRV samples masked (not imputed)")
    print(f"[{patient_id}] cleaned -> {dest}")
    return dest


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m pipeline.cleaning.cleaning <patient_id>")
        raise SystemExit(2)
    clean_patient(sys.argv[1])
