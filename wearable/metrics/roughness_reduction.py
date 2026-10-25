"""Reduction in mean absolute first difference — how much minute-to-minute jitter went."""
import numpy as np

KIND = "signal"
LABEL = "Roughness reduction %"
HIGHER_IS_BETTER = True


def compute(raw, smoothed):
    raw, smoothed = np.asarray(raw, dtype=float), np.asarray(smoothed, dtype=float)
    m = np.isfinite(raw) & np.isfinite(smoothed)
    if m.sum() < 3:
        return float("nan")
    r = np.mean(np.abs(np.diff(raw[m])))
    return float("nan") if r == 0 else float(100.0 * (1.0 - np.mean(np.abs(np.diff(smoothed[m]))) / r))
