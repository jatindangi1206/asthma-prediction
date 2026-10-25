"""Fraction of each raw peak's local excursion the smoothed signal retains.

The counterweight to variance_reduction: a smoother that flattens the volatility bursts
an exacerbation shows up as scores well on variance and badly here. Measured per
detected peak and reduced by median, so one huge spike cannot carry the score.
"""
import numpy as np
from scipy.signal import find_peaks

KIND = "signal"
LABEL = "Peak preservation (0-1)"
HIGHER_IS_BETTER = True
WINDOW = 6          # samples either side of a peak that define its local excursion


def compute(raw, smoothed, win: int = WINDOW):
    raw, smoothed = np.asarray(raw, dtype=float), np.asarray(smoothed, dtype=float)
    m = np.isfinite(raw) & np.isfinite(smoothed)
    if m.sum() < 3:
        return float("nan")
    raw, smoothed = raw[m], smoothed[m]

    peaks, _ = find_peaks(raw, prominence=0.5 * np.nanstd(raw))
    ratios = []
    for p in peaks:
        a, b = max(0, p - win), min(len(raw), p + win + 1)
        excursion = raw[p] - np.min(raw[a:b])
        if excursion > 1e-9:
            ratios.append(np.clip((smoothed[p] - np.min(smoothed[a:b])) / excursion, 0, 1))
    return float(np.median(ratios)) if ratios else float("nan")
