"""Brier score: mean squared error of the predicted probability. Calibration, not ranking."""
import numpy as np

KIND = "model"
LABEL = "Brier score"
HIGHER_IS_BETTER = False


def compute(y_true, y_prob):
    y_true = np.asarray(y_true).astype(float)
    return float(np.mean((np.asarray(y_prob, dtype=float) - y_true) ** 2))
