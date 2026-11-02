"""Forecasters: predict the next stretch of a signal from the stretch before it.

A pool of its own, not a corner of `smoothing/`, because the two answer different
questions and their residuals mean different things:

    smoothing     estimate the level AT t from data up to t        -> residual = noise
    forecasting   predict t+h from data up to t                    -> residual = SURPRISE

Putting a forecaster in `sweep smoothing` would compare two different quantities, and
the smoothing contract has nowhere to put a predictive distribution — which is the part
worth having. "This reading fell at the 99.5th percentile of what the model expected"
is calibrated; "this reading was 6 ms off" is not.

    forecast(contexts, horizon) -> (median, quantiles | None)
        contexts   (n, T) float, NaN where unobserved
        median     (n, horizon)
        quantiles  (n, horizon, len(LEVELS)) or None if the method is not probabilistic

`LEVELS` names the quantile levels a method returns, ascending. The `forecast` step
turns them into `tail_<signal>`, the observed value's position in the predictive
distribution.
"""
from __future__ import annotations

from wearable.registry import available, get, registry

POOL = "forecasting"
# The levels a probabilistic method should return if it can. Timestral and Chronos-2
# both emit exactly these 21; a method with fewer declares its own `LEVELS`.
LEVELS = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
          0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)


def names() -> list[str]:
    return available(POOL)


def all_forecasters() -> dict:
    return registry(POOL)


def forecaster(name: str):
    return get(POOL, name)
