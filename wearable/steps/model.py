"""Step: model. Trains on a join artifact's train table, predicts on its test table.

    input   a join artifact
    output  table `predictions`: p_id, date, y, y_prob

Takes whatever dataset it is given, so smoothing and CPD can both be skipped entirely —
`join.run(artifacts=[])` is a raw-feature baseline and this step treats it like any
other input.

Depends on no other step: it only needs the two tables and a model name.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wearable import models, registry, store
from wearable.config import SEED

STEP = "model"


def _xy(table: pd.DataFrame, feature_cols):
    X = table.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce")
    y = table["y"].astype(int).to_numpy()
    return X, y


def run(join_artifact, model: str = models.DEFAULT_MODEL, seed: int = SEED,
        force: bool = False, **overrides):
    spec = registry.get("models", model)
    params = {**spec.params(**overrides), "seed": seed}
    inputs = [join_artifact.ref]

    if not force:
        hit = store.find(STEP, model, params, inputs, spec.source_hash)
        if hit is not None:
            print(f"[model/{model}] reusing {hit.key[:12]}")
            return hit

    feature_cols = join_artifact.manifest.get("feature_cols") or []
    train = join_artifact.read_table("train")
    test = join_artifact.read_table("test")
    if not feature_cols:
        feature_cols = [c for c in train.columns if c not in {"p_id", "date", "y"}]

    Xtr, ytr = _xy(train, feature_cols)
    Xte, yte = _xy(test, feature_cols)
    if ytr.sum() == 0 or len(np.unique(ytr)) < 2:
        raise ValueError(f"train split has no positive class "
                         f"({ytr.sum()} positives of {len(ytr)}) — cannot fit")

    art = store.open_artifact(STEP, model, params, inputs, spec.source_hash,
                              label=spec.label)
    mod = spec.load()
    fitted = mod.fit(Xtr, ytr, seed=seed, **overrides)
    prob = np.asarray(mod.predict_proba(fitted, Xte), dtype=float)

    preds = test[["p_id", "date"]].copy()
    preds["y"] = yte
    preds["y_prob"] = prob
    art.write_table("predictions", preds)

    art.complete(n_train=len(Xtr), n_test=len(Xte), n_features=len(feature_cols),
                 train_positives=int(ytr.sum()), test_positives=int(yte.sum()),
                 mean_prob=float(prob.mean()))
    print(f"[model/{model}] fit on {len(Xtr):,} days, predicted {len(Xte):,} "
          f"(key {art.key[:12]})")
    return art
