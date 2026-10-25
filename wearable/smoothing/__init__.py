"""Smoothing pool. One file per method; drop one in and it is registered.

Contract — a module here is a smoothing method if it exposes:

    smooth_dataframe(df, patient_col, timestamp_col, value_col, config, out_col) -> DataFrame
        chunks per H8 with fresh state, runs causally, emits a strictly positive column

and optionally declares `LABEL`, `CONFIG` (its dataclass) and `DEFAULTS` (its knobs).
There is no list to edit — see `wearable/registry.py`.

`pf` (SMC + FFBS) is deliberately absent: its backward pass makes its output
non-causal, so it cannot satisfy the contract. It lives in `legacy/`.
"""
from wearable.registry import available, get, registry


def all_methods():
    return registry("smoothing")


def names():
    return available("smoothing")


def method(name):
    return get("smoothing", name)
