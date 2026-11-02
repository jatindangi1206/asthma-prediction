"""Never flags a day. The reference every other model has to beat.

With exacerbation days at ~1% of the cohort, "no" scores about 99% accuracy and a
specificity of 1.0 while catching nothing. Any comparison that reports accuracy without
this row is unreadable, so it is a model like any other and goes through the same
split, the same metrics and the same table.

It is also the floor for AUPRC: its average precision is the base rate itself.
"""
from __future__ import annotations

import numpy as np

LABEL = "Reference: never flags a day"
DEFAULTS: dict = {}


def fit(X, y, seed: int = 42, **params):
    return None


def predict_proba(fitted, X):
    return np.zeros(len(X), dtype=float)
