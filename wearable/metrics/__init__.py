"""Metric pool. One file per metric; drop one in and it is registered.

Contract — a module here is a metric if it exposes `compute` and declares `KIND`:

    KIND = "model"    compute(y_true, y_prob)  -> float   judges a prediction
    KIND = "signal"   compute(raw, smoothed)   -> float   judges a method, no model
    KIND = "event"    compute(predictions)     -> float   judges whole episodes

`model` and `signal` exist because a smoothing method can be judged either way: put it
through a model and compare prediction probabilities, or compare raw against smoothed
directly. `event` exists because an episode is caught when ANY of its days is flagged,
which the two arrays cannot express — so those metrics are handed the whole predictions
frame (`p_id, date, y, y_prob, event`, plus `group` when it is known).

`HIGHER_IS_BETTER` (default True) tells the comparison table which way to rank.

The metrics are deliberately not fixed — add a file and it appears in every comparison.
A metric whose name starts with `_` is a helper, not a metric (see `_confusion.py`).
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
