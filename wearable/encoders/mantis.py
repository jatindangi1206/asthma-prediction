"""MantisV2 (Feofanov et al., ICML'26), frozen — a representation per patient-day.

    python -m wearable encode mantis --signal hr --days 7 --profile 73

An **8.1M-parameter** encoder contrastively pretrained on synthetic series, built for
classification. `MantisTrainer.transform()` returns the embedding and that is all this
step wants. Three checkpoints exist (`Mantis-8M`, `MantisPlus`, `MantisV2`); `MantisV2`
is the one the paper's results use.

WHY IT IS WORTH A ROW: it is two orders of magnitude smaller than MOMENT, runs on CPU
in seconds, is Apache-2.0, and was designed for exactly this task rather than adapted to
it. If it matches MOMENT, the cheap model wins and the expensive one has nothing to add.

READ ITS RESULT WITH THIS IN MIND — it has **no missing-value mask.** Nothing in the
published API accepts one, so the holes in a window have to be filled before it sees
them, and `mode="interpolate"` (its own README's recommendation) is what does it. Two
consequences worth stating plainly, because they can manufacture a result:

  1. Interpolation invents values across gaps. On HRV that is two thirds of a window.
     On heart rate it is a few percent, which is why `--signal hr` is the honest way to
     run this one.
  2. Interpolating to a fixed length rescales time per row: a day with 20 readings and
     a day with 140 both become 512 points, so rate-of-change stops being comparable
     between rows. `mode="decimate"` avoids that at the cost of detail, and
     `windowing.bins_for` avoids needing either by making the window fit.

So if Mantis beats MOMENT here, suspect the resampling before believing the model. The
`mode` is part of the artifact key, so the two can be compared directly.

Code Apache-2.0 (the weights' licence is not stated on the model card — check before
anything but research). Lazily imported, not in `requirements.txt`, for the same reason
as the other foundation methods.

    pip install mantis-tsfm
"""
from __future__ import annotations

import numpy as np

from wearable.encoders._resample import fit_length, standardise

LABEL = "MantisV2 frozen representation (8.1M)"
CONTEXT = 512          # a multiple of 32 is required; 512 is what it was pretrained at
DEFAULTS = {
    "repo": "paris-noah/MantisV2",
    "mode": "interpolate",   # its own recommendation; see the caveat above
    "layer": 2,              # best frozen layer on UCR per the README (0-indexed)
    "batch": 256,
    "device": None,
}
_MODELS: dict = {}


def _model(repo: str, device, layer: int):
    """Load once per process, at first use — never at import time."""
    key = (repo, device, layer)
    if key not in _MODELS:
        try:
            import torch
            from mantis.architecture import MantisV2
            from mantis.trainer import MantisTrainer
        except ImportError as exc:
            raise ImportError("the `mantis` package is needed for the mantis encoder: "
                              "pip install mantis-tsfm") from exc
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # mantis-tsfm 1.1: the layer and output token are set on the network, not per call
        net = MantisV2(device=dev, return_transf_layer=layer,
                       output_token="combined").from_pretrained(repo)
        _MODELS[key] = MantisTrainer(device=dev, network=net)
    return _MODELS[key]


def encode(series, **params):
    """(n, T) window, NaN where unobserved -> (n, d) embeddings."""
    p = {**DEFAULTS, **{k: v for k, v in params.items() if k in DEFAULTS}}
    model = _model(p["repo"], p["device"], int(p["layer"]))

    x, mask = fit_length(series, CONTEXT, p["mode"])
    x = standardise(x, mask)
    out = []
    for k in range(0, len(x), int(p["batch"])):
        # (batch, channels, time); one channel, as everywhere in this repo
        xb = x[k:k + int(p["batch"])][:, None, :]
        z = model.transform(xb)
        out.append(np.asarray(z, dtype=np.float32).reshape(len(xb), -1))
    return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)
