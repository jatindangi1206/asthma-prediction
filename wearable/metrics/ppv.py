"""Share of flagged days that were exacerbation days (precision).

At a ~1% base rate this is the number a clinician feels: even a specific model raises
mostly false alarms, which is why it is reported beside sensitivity and never alone.
"""
from wearable.metrics._confusion import counts, div

KIND = "model"
LABEL = "PPV (precision)"
HIGHER_IS_BETTER = True


def compute(y_true, y_prob):
    tp, fp, _tn, _fn = counts(y_true, y_prob)
    return div(tp, tp + fp)
