"""Share of normal days left unflagged. The alarm-burden question."""
from wearable.metrics._confusion import counts, div

KIND = "model"
LABEL = "Specificity"
HIGHER_IS_BETTER = True


def compute(y_true, y_prob):
    _tp, fp, tn, _fn = counts(y_true, y_prob)
    return div(tn, tn + fp)
