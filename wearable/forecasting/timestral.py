"""Timestral-209M (Mistral), frozen — a causal baseline and its predictive distribution.

    python -m wearable forecast timestral --signal hr
    python -m wearable detect --on surprise --forecasting timestral --signal hr

A 209M patch transformer (24 layers, width 768, 32 steps per patch, context to 32,768)
trained for zero-shot forecasting. Here it is asked, block by block, what the next
stretch of the signal should be; `_base.py` does the windowing and causality.

THE PICK OF THE THREE. It is the only one of them whose published API names a
missing-value marker — NaN, plus an `observation_mask` — so the holes in this cohort's
series are declared rather than filled. It returns 21 quantiles, which is what makes
`tail_<sig>` possible. Modified MIT, far less restrictive than TimesFM 3's
non-commercial terms. And on the public forecasting benchmarks it reportedly beats
TimesFM 3.

Frozen, so there is nothing fitted on the cohort and no train/test leak to argue about —
see `_base.py` on why that makes it truncation-exact where nine of ten smoothers are not.

Lazily imported and deliberately not in `requirements.txt`: an unimportable method fails
`selftest`, and a ~0.8 GB download has no business in an architecture check.

    pip install git+https://github.com/mistralai/timestral.git   # py>=3.10, torch>=2.4
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wearable.forecasting import LEVELS
from wearable.forecasting._base import ForecastConfig

LABEL = "Timestral-209M zero-shot forecast (frozen)"
_MODELS: dict = {}


@dataclass
class TimestralConfig(ForecastConfig):
    repo: str = "mistralai/Timestral-209M"   # 5M / 28M / 123M / 209M all exist
    device: str | None = None                # None = cuda when available, else cpu


CONFIG = TimestralConfig
DEFAULTS = {"repo": "mistralai/Timestral-209M", "context": 512, "stride": 32,
            "min_context": 64}


def _model(cfg):
    """Load once per process, at first use — never at import time."""
    key = (cfg.repo, cfg.device)
    if key not in _MODELS:
        try:
            import torch
            from timestral import TimestralForecaster
        except ImportError as exc:
            raise ImportError(
                "the `timestral` package is needed for the timestral forecaster: "
                "pip install git+https://github.com/mistralai/timestral.git") from exc
        dev = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        _MODELS[key] = TimestralForecaster.from_pretrained(cfg.repo, device=dev)
    return _MODELS[key]


def forecast(contexts, horizon, config=None, levels=LEVELS):
    """(n, T) contexts with NaN for an unobserved slot -> (median, quantiles).

    NaN is Timestral's own missing marker, so the holes are declared, not filled.
    `median` comes back (batch, variates, horizon) and there is one variate; `quantiles`
    comes back (batch, variates, horizon, 21).
    """
    cfg = config or TimestralConfig()
    out = _model(cfg).forecast(np.asarray(contexts, dtype=np.float32),
                              prediction_length=int(horizon))
    n, h = len(contexts), int(horizon)
    host = lambda a: np.asarray(a.detach().cpu() if hasattr(a, "detach") else a, dtype=float)
    median = host(out.median).reshape(n, h)          # tensors come back on the GPU
    try:
        quant = host(out.quantiles).reshape(n, h, -1)
    except Exception:          # a build without the quantile array: median only
        quant = None
    return median, quant
