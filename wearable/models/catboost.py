"""Gradient-boosted trees (CatBoost) on the day window — the reference design's 6.3.

The second tree model of the comparison, with the reference design's settings: 400
iterations, depth 4, learning rate 0.05, classes balanced automatically.

It earns a row next to `xgboost` because it builds its trees differently — oblivious
trees (every node at one depth splits on the same feature) grown with ordered boosting,
which is less prone to overfitting a small positive class than standard boosting. With
144 exacerbation days in the whole cohort that difference is the interesting one.

Writes nothing to disk (`allow_writing_files=False`): the default leaves `catboost_info/`
behind in the working directory, and a result's only record in this repo is its
artifact manifest.
"""
from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier

LABEL = "CatBoost on day features"
DEFAULTS = {"iterations": 400, "depth": 4, "learning_rate": 0.05,
            "auto_class_weights": "Balanced"}


def fit(X, y, seed: int = 42, **params):
    p = {**DEFAULTS, **{k: v for k, v in params.items() if k in DEFAULTS}}
    model = CatBoostClassifier(**p, random_seed=seed, verbose=False,
                               allow_writing_files=False)
    model.fit(np.asarray(X, dtype=float), np.asarray(y).astype(int))
    return model


def predict_proba(fitted, X):
    return fitted.predict_proba(np.asarray(X, dtype=float))[:, 1]
