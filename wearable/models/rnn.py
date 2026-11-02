"""Recurrent network over the day window — the reference design's 6.4.

A plain forward RNN reads the window one day at a time, oldest first, and the state at
the classified day feeds one logistic output. Everything else — the "day present" flags,
the scaling, the half-exacerbation batches, the fixed step count — is in `_seq.py`,
which `lstm.py` shares.

An RNN rather than an LSTM is the ablation worth having: with a 7-day window the
question is whether a gate that can hold state for weeks buys anything over plain
recurrence over a week.
"""
from __future__ import annotations

from wearable.models import _seq

LABEL = "RNN over the day window"
DEFAULTS = dict(_seq.DEFAULTS)
CELL = "RNN"


def fit(X, y, seed: int = 42, **params):
    return _seq.fit(X, y, CELL, seed=seed, **params)


def predict_proba(fitted, X):
    return _seq.predict_proba(fitted, X)
