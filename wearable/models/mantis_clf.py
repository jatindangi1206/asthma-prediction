"""MantisV2 as the classifier itself — a foundation model doing the prediction.

    python -m wearable sweep inputs --variants series --model_names mantis_clf,xgboost

Every other foundation route in this repo uses a pretrained model in the MIDDLE: the
forecasters turn a signal into surprise, the encoders turn a window into a vector, and a
small model does the predicting. This file is the other arrangement — the pretrained
model reads the raw series and emits the probability.

WHY MANTIS AND NOT MOMENT
-------------------------
Because of the only number that matters here: **144 positive days in the whole cohort.**

MantisV2 is **8.1M parameters**. MOMENT-base is ~110M and MOMENT-large ~385M. Against
144 positives, freezing a 385M body and training a linear head is not "using the model
directly" at all — it is exactly what `encoders/moment.py` already does, with a weaker
head, since `join` + a tree model can put anything on top of an embedding. Mantis is the
one model on the shortlist small enough that training it end to end on this data is a
real option rather than a memorisation exercise.

The second reason is **cross-channel structure**. Mantis takes several channels at once
and can learn a combination of them (`LinearChannelCombiner`, trained with the head).
Frozen per-channel embeddings concatenated into one table cannot do that — each channel
is embedded in ignorance of the others. So this is the one route that can find a
relationship *between* HR and HRV rather than within either.

WHAT TO EXPECT — read the result against this
---------------------------------------------
The reference COPD design ran the equivalent experiment and reported, on its own tables,
TimesFM 3 frozen with a learned logistic layer at onset AUROC 0.552 [0.44-0.66] and
detection AUROC 0.499 — chance — while its purpose-built end-to-end network (GAST)
managed 0.506 and was beaten by its own ablation. Gradient-boosted trees were among the
better rows. Pretraining transfers knowledge of what time series look like; it transfers
no knowledge of what an exacerbation looks like, and that mapping has to be learned from
the 144.

So this file exists to answer the question, not because the answer is expected to be yes.
A clean negative is worth reporting: readers will ask whether a foundation model can
simply predict this, and "we tried, here is the number" is the answer.

INPUT
-----
The raw series, which means profile columns (`join --profile`). `_seq.sequence` recovers
the window from the column names, so this file needs no new contract: `<feature>_p<slot>`
columns become one channel each, and a day is one sample of shape (channels, length).
Mantis wants the length a multiple of 32 and was pretrained at 512, so
`encoders/_resample` fits the window to that — the mode is a keyed parameter, because
how a short window is stretched to 512 is a research choice and not a detail.

Licence: code Apache-2.0; the weights' licence is not stated on the model card, so treat
it as research-only until checked. Lazily imported and not in `requirements.txt`, like
every other foundation method here.

    pip install mantis-tsfm
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wearable.encoders._resample import fit_length, standardise
from wearable.windowing import channel_series

LABEL = "MantisV2 as the classifier (8.1M, end to end)"
CONTEXT = 512          # a multiple of 32 is required; 512 is what it was pretrained at
DEFAULTS = {
    "repo": "paris-noah/MantisV2",
    "mode": "interpolate",      # how the window is fitted to CONTEXT
    # `full` trains the whole 8.1M body — which IS the experiment: can a pretrained
    # model learn this task end to end? `head` freezes the body and trains only the
    # classifier, which is the conservative reading and what a 385M model would be
    # limited to. Keyed, so the two are comparable.
    "fine_tuning_type": "full",
    "epochs": 20,               # the trainer defaults to 500; 144 positives do not need it
    "batch": 64,
    "base_learning_rate": 2e-4,
    "channels": 5,              # channels the adapter reduces to, when there are more
    "device": None,
}


def _trainer(p: dict):
    """Build a fresh trainer. Loaded at first use, never at import time."""
    try:
        import torch
        from mantis.architecture import MantisV2
        from mantis.trainer import MantisTrainer
    except ImportError as exc:
        raise ImportError("the `mantis` package is needed for mantis_clf: "
                          "pip install mantis-tsfm") from exc
    dev = p["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    net = MantisV2(device=dev).from_pretrained(p["repo"])
    return MantisTrainer(device=dev, network=net), dev


def _tensor(X, p: dict) -> np.ndarray:
    """A modelling table -> (samples, channels, CONTEXT).

    One channel per profiled feature. A day with no reading in a slot is a hole, and
    `fit_length` is what decides how it is filled — see `encoders/_resample`.
    """
    names, v = channel_series(pd.DataFrame(X))   # (n, channels, time)
    n, c, t = v.shape
    if t < 32:
        raise ValueError(
            f"mantis_clf reads the recorded series, not day summaries: got {t} points "
            f"per channel. Build the table with `--profile 55 --days 7` or finer "
            f"(see wearable/windowing.py on the context budget)")
    out = np.empty((n, c, int(CONTEXT)), dtype=np.float32)
    for j in range(c):
        x, mask = fit_length(v[:, j, :], CONTEXT, p["mode"])
        out[:, j] = standardise(x, mask)
    return out


def fit(X, y, seed: int = 42, **params):
    p = {**DEFAULTS, **{k: v for k, v in params.items() if k in DEFAULTS}}
    import torch
    torch.manual_seed(seed)

    model, dev = _trainer(p)
    x = _tensor(X, p)
    y = np.asarray(y).astype(int)
    if y.sum() == 0:
        raise ValueError("no exacerbation day among the training days")

    adapter = None
    if x.shape[1] > int(p["channels"]):
        # Too many channels for the pretrained body. A LEARNABLE combiner is trained
        # with the head, so the model chooses how to mix the channels — which is the
        # whole point of asking it for cross-channel structure. A fixed PCA projector
        # would decide that in advance.
        from mantis.adapters import LinearChannelCombiner
        adapter = LinearChannelCombiner(num_channels=x.shape[1],
                                        new_num_channels=int(p["channels"]))

    kw = dict(num_epochs=int(p["epochs"]), batch_size=int(p["batch"]),
              base_learning_rate=float(p["base_learning_rate"]),
              fine_tuning_type=p["fine_tuning_type"])
    if adapter is not None:
        # With a channel adapter the trainer wants the adapter trained alongside.
        kw.update(adapter=adapter, fine_tuning_type="adapter_head")
    model.fit(x, y, **kw)
    return {"model": model, "params": p, "cols": list(pd.DataFrame(X).columns)}


def predict_proba(fitted, X):
    p = fitted["params"]
    x = _tensor(pd.DataFrame(X).reindex(columns=fitted["cols"]), p)
    out = np.asarray(fitted["model"].predict_proba(x), dtype=float)
    return out[:, 1] if out.ndim == 2 and out.shape[1] > 1 else out.ravel()
