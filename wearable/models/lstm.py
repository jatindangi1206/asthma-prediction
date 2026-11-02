"""LSTM over the day window — the reference design's 6.4.

`rnn.py` with the recurrent cell swapped: the gates let the network carry a slow drift
across the whole window instead of having it decay step by step, which is the
H2/H4 drifting-baseline story told by a model rather than by a smoother.

One file, one cell name — the shared encoder, training loop and masking live in
`_seq.py`, so this and `rnn.py` differ in exactly the thing being compared.
"""
from __future__ import annotations

from wearable.models import _seq

LABEL = "LSTM over the day window"
DEFAULTS = dict(_seq.DEFAULTS)
CELL = "LSTM"


def fit(X, y, seed: int = 42, **params):
    return _seq.fit(X, y, CELL, seed=seed, **params)


def predict_proba(fitted, X):
    return _seq.predict_proba(fitted, X)
