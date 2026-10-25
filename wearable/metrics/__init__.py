"""Metric pool. One file per metric; drop one in and it is registered.

Contract — a module here is a metric if it exposes `compute` and declares `KIND`:

    KIND = "model"    compute(y_true, y_prob) -> float    judges a prediction
    KIND = "signal"   compute(raw, smoothed)  -> float    judges a method, no model

The two kinds exist because a smoothing method can be judged either way: put it through
a model and compare prediction probabilities, or compare raw against smoothed directly.
`HIGHER_IS_BETTER` (default True) tells the comparison table which way to rank.

The metrics are deliberately not fixed — add a file and it appears in every comparison.
"""
from wearable.registry import available, get, registry


def all_metrics(kind: str | None = None):
    reg = registry("metrics")
    return {n: m for n, m in reg.items() if kind is None or m.kind == kind}


def names(kind: str | None = None):
    return [n for n in available("metrics")
            if kind is None or registry("metrics")[n].kind == kind]


def metric(name):
    return get("metrics", name)


def higher_is_better(name: str) -> bool:
    return bool(getattr(get("metrics", name).load(), "HIGHER_IS_BETTER", True))
