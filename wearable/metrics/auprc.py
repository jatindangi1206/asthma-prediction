"""Average precision. The honest headline when positives are ~1% of days."""
import numpy as np
from sklearn.metrics import average_precision_score

KIND = "model"
LABEL = "AUPRC (average precision)"
HIGHER_IS_BETTER = True


def compute(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))
