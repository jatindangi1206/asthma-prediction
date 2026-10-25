"""The H8 segmentation rule — the one definition for the repo."""
from __future__ import annotations

import pandas as pd

from wearable.config import GAP_THRESHOLD_MIN


def assign_chunks(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    patient_col: str = "patient_id",
    gap_threshold_min: float = GAP_THRESHOLD_MIN,
) -> pd.DataFrame:
    """Add `time_diff_min`, `gap_boundary`, `chunk_id`; sorted by (patient, time).

    A gap >= the threshold opens a new chunk, whose first row is t=0 for the filter
    (H8 state reset). Sub-threshold gaps stay in the same chunk and are handled
    downstream as irregular sampling (H7). ids start at 1: a patient's first row has a
    NaN gap, which counts as a boundary.
    """
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col])
    out = out.sort_values([patient_col, timestamp_col]).reset_index(drop=True)

    dt = out.groupby(patient_col, sort=False)[timestamp_col].diff().dt.total_seconds() / 60.0
    boundary = dt.isna() | (dt >= gap_threshold_min)
    out["time_diff_min"] = dt
    out["gap_boundary"] = boundary
    out["chunk_id"] = boundary.cumsum().astype(int)
    return out
