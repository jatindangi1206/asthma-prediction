"""Encoders: a window of signal -> a vector per patient-day.

A pool of its own because the output shape is unlike anything else here. A smoother
emits one value per reading, a detector one flag per reading, a model one probability
per day — an encoder emits a fixed-width VECTOR per day, which is a feature block for
whatever model comes next, not a prediction.

    encode(series) -> (n, d)      series: (n, T) float, NaN where unobserved

`CONTEXT` is the model's fixed input length when it has one (MOMENT: 512). The `encode`
step resamples the window onto it, so a method never has to do its own padding.

Adding one is one file here. See `wearable/windowing.py` for the context budget: the
window the step hands over is `days * profile` slots wide.
"""
from __future__ import annotations

from wearable.registry import available, get, registry

POOL = "encoders"


def names() -> list[str]:
    return available(POOL)


def all_encoders() -> dict:
    return registry(POOL)


def encoder(name: str):
    return get(POOL, name)
