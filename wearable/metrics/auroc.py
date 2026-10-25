"""Area under the ROC curve. Threshold-free ranking quality."""
import numpy as np
from sklearn.metrics import roc_auc_score

KIND = "model"
LABEL = "AUROC"
HIGHER_IS_BETTER = True


def compute(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:      # undefined with one class present
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))
