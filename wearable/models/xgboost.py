"""Gradient-boosted trees (XGBoost) on the day window — the reference design's 6.2.

Settings are the reference design's, unchanged, so a row here is comparable with the
one it was ported from: 400 shallow trees, learning rate 0.05, and both the rows and
the columns subsampled to 0.8 so the trees decorrelate.

Missing values are not imputed: XGBoost sends them down a learned default branch at
every split. That matters more here than the exact tree settings — a wearable day is
often partly missing, and *which* days are missing is itself informative (charging,
removal, non-compliance). That is the property the reference design wanted from a tree model, and
the reason this is the headline one here.

Exacerbation days are ~1% of days, so the positive class is weighted up by the ratio of
normal to exacerbation days (`scale_pos_weight`), which is the reference's rule and the
equivalent of `class_weight="balanced"` elsewhere in this pool.
"""
from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier

LABEL = "XGBoost on day features"
DEFAULTS = {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist"}


def fit(X, y, seed: int = 42, **params):
    p = {**DEFAULTS, **{k: v for k, v in params.items() if k in DEFAULTS}}
    y = np.asarray(y).astype(int)
    pos = int(y.sum())
    # 8 threads: on a ~12k-row table every core is overhead (40 threads: 56 s, 8: 3 s)
    model = XGBClassifier(**p, random_state=seed, eval_metric="logloss", n_jobs=8,
                          scale_pos_weight=(len(y) - pos) / max(pos, 1))
    model.fit(np.asarray(X, dtype=float), y)
    return model


def predict_proba(fitted, X):
    return fitted.predict_proba(np.asarray(X, dtype=float))[:, 1]
