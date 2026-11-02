"""Model pool. One file per model; drop one in and it is registered.

Contract — a module here is a model if it exposes:

    fit(X, y, seed, **params) -> fitted
    predict_proba(fitted, X)  -> 1-D array of P(y=1)

and optionally declares `LABEL` and `DEFAULTS`. X is a numeric DataFrame, y a 0/1
array. Handle NaN internally: a wearable day-feature matrix is always partly missing.
There is no list to edit — see `wearable/registry.py`.
"""
from wearable.registry import available, get, registry

# The cheapest of the real models, so a sweep that does not name one still runs
# something comparable rather than the reference.
DEFAULT_MODEL = "xgboost"


def all_methods():
    return registry("models")


def names():
    return available("models")


def method(name):
    return get("models", name)
