"""Confusion counts at the fixed flag threshold — shared by the threshold metrics.

A module whose name starts with `_` is skipped by the registry, so this is a helper and
not a metric. Every metric below it answers one question about the same four counts, at
`config.THRESHOLD`: a probability at or above it is a flagged day.
"""
from __future__ import annotations

import numpy as np

from wearable.config import THRESHOLD


def counts(y_true, y_prob, threshold: float = THRESHOLD):
    """-> (tp, fp, tn, fn) at `threshold`."""
    y = np.asarray(y_true).astype(int)
    flag = np.asarray(y_prob, dtype=float) >= threshold
    return (int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum()),
            int((~flag & (y == 0)).sum()), int((~flag & (y == 1)).sum()))


def div(num, den) -> float:
    """A rate that is NaN rather than an error when nothing was counted."""
    return float(num) / den if den else float("nan")
