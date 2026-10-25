"""How far the smoothed mean drifts from the raw mean, as a percentage. Bias, not spread."""
import numpy as np

KIND = "signal"
LABEL = "Mean error %"
HIGHER_IS_BETTER = False


def compute(raw, smoothed):
    raw, smoothed = np.asarray(raw, dtype=float), np.asarray(smoothed, dtype=float)
    m = np.isfinite(raw) & np.isfinite(smoothed)
    if m.sum() < 3 or np.mean(raw[m]) == 0:
        return float("nan")
    return float(100.0 * abs(np.mean(smoothed[m]) - np.mean(raw[m])) / abs(np.mean(raw[m])))
