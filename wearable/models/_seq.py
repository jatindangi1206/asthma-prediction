"""Shared machinery for the models that read the day window as a SEQUENCE.

Not a method itself — the leading underscore keeps it out of the registry. `rnn.py` and
`lstm.py` are the same network with the recurrent cell changed, which is how the
reference design carried them too (its 6.4).

Three things live here:

`sequence()` recovers the window from the column names. `join` names a windowed column
`<feature>_d<k>` for k days before the classified day, so a flat day-feature frame
turns back into (samples, days, features) with the oldest day first — and no new
contract between the steps is needed for it. An unwindowed frame (`join`'s `days=1`) is
simply one time step, so these models run on any window including none.

The input is the scaled values **and a "day present" flag per feature**, exactly as the
reference fed its networks the readings plus their presence masks. A day with no
recording is a real observation about a patient, so it is told to the network rather
than imputed; the flags are what let it tell "resting HRV of zero" from "watch on the
charger".

`train()` is the reference's loop with its early stopping removed. The reference held
back *patients* to early-stop on, and this repo's model contract is handed features and
a label and never sees `p_id` — a day-level holdout would put the same patient on both
sides and early-stop on leaked information, which is worse than not early-stopping. So
the step count is fixed (`DEFAULTS["steps"]`) and every other part of the loop is the
reference's: batches half exacerbation days, Adam, BCE on the logit, gradients clipped.

Sizing note, not a bug: a 7-day window is 7 time steps and the cohort has 144
exacerbation days. The reference ran these networks on 10,080 minutes x 7 signals per
sample, where a recurrent encoder has something to compress. Here they are a fair row in
the comparison, not the favourite — read them next to `xgboost` and `catboost`.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
import torch

# The reference design's network settings (its 6.4), less the early-stopping ones.
DEFAULTS = {"hidden": 64, "layers": 2, "dropout": 0.1,
            "steps": 600, "batch": 32, "lr": 1e-3, "clip": 1.0}


def sequence(X: pd.DataFrame) -> np.ndarray:
    """A day-feature frame -> (n, days, features), oldest day first, NaN where missing."""
    cols: dict[str, dict[int, str]] = {}
    for c in X.columns:
        m = re.fullmatch(r"(.*)_d(\d+)", str(c))
        base, k = (m.group(1), int(m.group(2))) if m else (str(c), 0)
        cols.setdefault(base, {})[k] = c
    bases = sorted(cols)
    T = max(max(ks) for ks in cols.values()) + 1 if bases else 1
    v = np.full((len(X), T, len(bases)), np.nan, dtype=np.float32)
    for j, base in enumerate(bases):
        for k, c in cols[base].items():
            # k days back; k=0 (the classified day) is the LAST step.
            v[:, T - 1 - k, j] = pd.to_numeric(X[c], errors="coerce").to_numpy(np.float32)
    return v


def scales(v: np.ndarray) -> np.ndarray:
    """Each feature's largest absolute value in the training days — a change of units
    only, as in the reference. A feature that is entirely missing or flat at zero gets
    1.0 so it cannot divide by zero."""
    with warnings.catch_warnings():
        # a feature with no reading anywhere is an all-NaN slice, handled on the next
        # line — nanmax warns about it through the warnings module, not errstate
        warnings.simplefilter("ignore", RuntimeWarning)
        s = np.nanmax(np.abs(v), axis=(0, 1))
    s = np.where(np.isfinite(s) & (s > 0), s, 1.0)
    return s.astype(np.float32)


def tensor(v: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """(n, T, F) raw -> (n, T, 2F): scaled values, then a flag per feature per day."""
    return np.concatenate([np.nan_to_num(v / scale),
                           (~np.isnan(v)).astype(np.float32)], axis=-1)


class Net(torch.nn.Module):
    """A recurrent encoder over the day window, read at the classified day."""

    def __init__(self, n_in: int, cell: str, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.rnn = getattr(torch.nn, cell)(
            n_in, hidden, layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0)      # torch warns on dropout w/ 1 layer
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        # The last step is the classified day, so that is where the state is read.
        return self.head(self.rnn(x)[0][:, -1]).squeeze(-1)


def fit(X, y, cell: str, seed: int = 42, **params):
    """Build and train one recurrent model; returns everything `predict` needs."""
    p = {**DEFAULTS, **{k: v for k, v in params.items() if k in DEFAULTS}}
    cols = list(X.columns)
    v = sequence(X)
    scale = scales(v)
    x = tensor(v, scale)
    y = np.asarray(y).astype(int)
    if y.sum() == 0:
        raise ValueError("no exacerbation day among the training days")

    torch.manual_seed(seed)
    # a 64-wide net gains nothing from 40 threads; with parallel sweeps they thrash (3 s -> 20 min)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    net = Net(x.shape[-1], cell, p["hidden"], p["layers"], p["dropout"])
    opt = torch.optim.Adam(net.parameters(), lr=p["lr"])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    xt, yt = torch.as_tensor(x), torch.as_tensor(y.astype(np.float32))

    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_pos = max(int(p["batch"]) // 2, 1)               # half the batch is positive
    n_neg = max(int(p["batch"]) - n_pos, 1)
    net.train()
    for _ in range(int(p["steps"])):
        idx = np.concatenate([rng.choice(pos, n_pos, replace=len(pos) < n_pos),
                              rng.choice(neg, n_neg, replace=len(neg) < n_neg)])
        opt.zero_grad()
        loss = loss_fn(net(xt[idx]), yt[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), p["clip"])
        opt.step()
    net.eval()
    return {"net": net, "scale": scale, "cols": cols}


def predict_proba(fitted, X, batch: int = 256):
    """Scaled and masked the same way as training, with the training columns' order."""
    x = tensor(sequence(pd.DataFrame(X).reindex(columns=fitted["cols"])),
               fitted["scale"])
    net = fitted["net"]
    net.eval()
    with torch.no_grad():
        out = [torch.sigmoid(net(torch.as_tensor(x[k:k + batch]))).cpu().numpy()
               for k in range(0, len(x), batch)]
    return np.concatenate(out) if out else np.zeros(0, dtype=float)
