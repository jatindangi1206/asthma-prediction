"""Pearson correlation between raw and smoothed. Did the baseline track the signal at all?"""
import numpy as np

KIND = "signal"
LABEL = "Correlation (raw vs smoothed)"
HIGHER_IS_BETTER = True


def compute(raw, smoothed):
    raw, smoothed = np.asarray(raw, dtype=float), np.asarray(smoothed, dtype=float)
    m = np.isfinite(raw) & np.isfinite(smoothed)
    if m.sum() < 3 or np.std(raw[m]) == 0 or np.std(smoothed[m]) == 0:
        return float("nan")
    return float(np.corrcoef(raw[m], smoothed[m])[0, 1])
