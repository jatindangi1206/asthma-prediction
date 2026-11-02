"""MOMENT (Auton Lab, CMU), frozen — a representation per patient-day.

    python -m wearable encode moment --signal hr --days 7 --profile 73

A T5-family encoder pretrained on the Time-series PILE, used here with
`task_name="embedding"`: the window goes in, a fixed-width vector comes out, and
whatever model follows is trained on that vector. Nothing is fine-tuned.

WHY MOMENT IS THE ONE TO TRY FIRST
----------------------------------
It takes an **`input_mask`** marking which timesteps hold a real reading. Of the
representation models on the shortlist it is the only one that does, and this cohort's
windows are mostly holes — a median HRV day has 54 of 144 ten-minute slots. Every other
option has to invent the missing two thirds before it can embed anything.

CONTEXT IS FIXED AT 512. Not a preference — its positional embeddings are pretrained at
that length, so `_resample` fits the window to it, and `wearable/windowing.bins_for`
picks a `--profile` that fits without anything lossy. At heart rate's ~744 readings a
day a 7-day window is 5,208 slots, so `--profile 73` (511 slots) is the setting that
keeps one slot meaning one fixed duration. On HRV a 7-day window is ~385 slots and pads
into 512 with nothing invented at all.

PREFER THE SMALL OR BASE CHECKPOINT. `MOMENT-1-large` emits `d=1024` against 144
exacerbation days in the whole cohort — seven features per event. The step keeps the
full vector and reduction happens inside the model, where the train/test boundary
exists; doing it here would fit on test patients. A smaller checkpoint is the cheaper
fix.

Licence MIT. Lazily imported and deliberately not in `requirements.txt`: an unimportable
method fails `selftest`, and a multi-gigabyte download has no business in the path of an
architecture check.

    pip install momentfm
"""
from __future__ import annotations

import numpy as np

from wearable.encoders._resample import fit_length, standardise

LABEL = "MOMENT frozen representation"
CONTEXT = 512          # fixed by the pretrained positional embeddings
DEFAULTS = {
    "repo": "AutonLab/MOMENT-1-base",   # small / base / large all exist
    "mode": "pad",                      # how the window is fitted to CONTEXT
    "reduction": "mean",                # over patches; "none" keeps every patch
    "batch": 64,
    "device": None,
}
_MODELS: dict = {}


def _model(repo: str, device):
    """Load once per process, at first use — never at import time."""
    key = (repo, device)
    if key not in _MODELS:
        try:
            import torch
            from momentfm import MOMENTPipeline
        except ImportError as exc:
            raise ImportError("the `momentfm` package is needed for the moment "
                              "encoder: pip install momentfm") from exc
        model = MOMENTPipeline.from_pretrained(
            repo, model_kwargs={"task_name": "embedding"})
        model.init()
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _MODELS[key] = (model.to(dev).eval(), dev)
    return _MODELS[key]


def encode(series, **params):
    """(n, T) window, NaN where unobserved -> (n, d) embeddings."""
    import torch
    p = {**DEFAULTS, **{k: v for k, v in params.items() if k in DEFAULTS}}
    model, dev = _model(p["repo"], p["device"])

    x, mask = fit_length(series, CONTEXT, p["mode"])
    x = standardise(x, mask)
    out = []
    with torch.no_grad():
        for k in range(0, len(x), int(p["batch"])):
            xb = torch.as_tensor(x[k:k + int(p["batch"])], device=dev)
            mb = torch.as_tensor(mask[k:k + int(p["batch"])].astype(np.float32),
                                 device=dev)
            # (batch, channels, time) — one channel: this repo models one signal at a
            # time, and each channel has its own cadence and its own gaps.
            res = model(x_enc=xb.unsqueeze(1), input_mask=mb)
            emb = getattr(res, "embeddings", res)
            out.append(np.asarray(emb.reshape(len(xb), -1).cpu(), dtype=np.float32))
    return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)
