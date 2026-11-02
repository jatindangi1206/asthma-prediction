"""Share of exacerbation days that were flagged (recall). The early-warning question."""
from wearable.metrics._confusion import counts, div

KIND = "model"
LABEL = "Sensitivity (recall)"
HIGHER_IS_BETTER = True


def compute(y_true, y_prob):
    tp, _fp, _tn, fn = counts(y_true, y_prob)
    return div(tp, tp + fn)
