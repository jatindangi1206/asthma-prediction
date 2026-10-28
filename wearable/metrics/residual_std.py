"""Spread of what the smoother removed: std(raw - smoothed), in the signal's units.

This residual is what `detect --on residual` consumes. Not monotone-good: 0 means
nothing was smoothed, so read it next to peak_preservation and correlation.
"""
import numpy as np

KIND = "signal"
LABEL = "Residual std"
HIGHER_IS_BETTER = False


def compute(raw, smoothed):
    raw, smoothed = np.asarray(raw, dtype=float), np.asarray(smoothed, dtype=float)
    m = np.isfinite(raw) & np.isfinite(smoothed)
    if m.sum() < 3:
        return float("nan")
    return float(np.std(raw[m] - smoothed[m]))
