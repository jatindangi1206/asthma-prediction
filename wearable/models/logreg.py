"""Logistic regression on day features — the comparison baseline.

Deliberately cheap: a method sweep runs 11 smoothing variants through the same model,
so the model must not dominate the runtime. Median-imputes, standardises, and balances
the classes (exacerbation days are ~1% of days).

Fully seeded, so a difference between two runs is the method, never the model.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LABEL = "Logistic regression on day features"
DEFAULTS = {"C": 1.0, "max_iter": 2000, "class_weight": "balanced"}


def fit(X, y, seed: int = 42, **params):
    p = {**DEFAULTS, **params}
    # Drop all-NaN columns up front: a variant can leave a feature entirely empty, and
    # the imputer would emit NaN for it and poison the fit.
    keep = [c for c in X.columns if X[c].notna().any()]
    pipe = Pipeline([
        ("prep", ColumnTransformer(
            [("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                               ("scale", StandardScaler())]), keep)],
            remainder="drop")),
        ("clf", LogisticRegression(C=p["C"], max_iter=p["max_iter"],
                                   class_weight=p["class_weight"],
                                   random_state=seed)),
    ])
    pipe.fit(X, np.asarray(y).astype(int))
    pipe._keep = keep
    return pipe


def predict_proba(fitted, X):
    return fitted.predict_proba(X)[:, 1]
