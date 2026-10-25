"""How much variance the smoother removed, as a percentage of the raw variance.

Model-free: high means a calm baseline, but 100% would mean the signal is gone, so read
it next to peak_preservation.
"""
import numpy as np

KIND = "signal"
LABEL = "Variance reduction %"
HIGHER_IS_BETTER = True


def compute(raw, smoothed):
    raw, smoothed = np.asarray(raw, dtype=float), np.asarray(smoothed, dtype=float)
    m = np.isfinite(raw) & np.isfinite(smoothed)
    if m.sum() < 3:
        return float("nan")
    v = np.var(raw[m])
    return float("nan") if v == 0 else float(100.0 * (1.0 - np.var(smoothed[m]) / v))
