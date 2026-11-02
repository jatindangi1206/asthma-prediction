"""Chronos-2 (Amazon), frozen — the covariate-aware forecaster.

    python -m wearable forecast chronos --signal hrv

120M parameters (`amazon/chronos-2`; `chronos-2-small` is 28M), probabilistic, with
quantile output. `_base.py` does the windowing and causality.

WHY IT EARNS A ROW THE OTHER TWO DO NOT: Chronos-2 takes **past and known-future
covariates**. Now that ingest carries every channel, that makes one genuinely new
clinical question askable — *does HRV become unpredictable GIVEN heart rate before an
exacerbation?* That is a question about decoupling between two channels, and no
single-channel method can ask it.

That is not what this file does yet. The covariate path needs a second channel aligned
onto the same grid, which is a change to `_base.predict_series` (its context is one
array), not to this adapter. Until then this is the third univariate benchmark, and the
covariate version is the next step — deliberately left undone rather than half-built,
because a covariate forecaster that silently ignores its covariates is worse than none.

Licence: the repo's licence section says Apache-2.0 while its badge says MIT. Check
`LICENSE` in the checkout before relying on either.

    pip install chronos-forecasting
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wearable.forecasting import LEVELS
from wearable.forecasting._base import ForecastConfig

LABEL = "Chronos-2 zero-shot forecast (frozen)"
_MODELS: dict = {}


@dataclass
class ChronosConfig(ForecastConfig):
    repo: str = "amazon/chronos-2"      # or autogluon/chronos-2-small (28M)
    device: str | None = None


CONFIG = ChronosConfig
DEFAULTS = {"repo": "amazon/chronos-2", "context": 512, "stride": 32,
            "min_context": 64}


def _model(cfg):
    """Load once per process, at first use — never at import time."""
    key = (cfg.repo, cfg.device)
    if key not in _MODELS:
        try:
            import torch
            from chronos import Chronos2Pipeline
        except ImportError as exc:
            raise ImportError(
                "the `chronos-forecasting` package is needed for the chronos "
                "forecaster: pip install chronos-forecasting") from exc
        dev = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        _MODELS[key] = Chronos2Pipeline.from_pretrained(cfg.repo, device_map=dev)
    return _MODELS[key]


def forecast(contexts, horizon, config=None, levels=LEVELS):
    """(n, T) contexts -> (median, quantiles). The only vendor call.

    Passed as a list of 1-D arrays of the readings each context actually holds: the
    predict_df surface wants a long frame with an id column, and the array surface is
    the closer fit to a batch of equal-horizon windows.
    """
    cfg = config or ChronosConfig()
    h = int(horizon)
    series = []
    for c in np.asarray(contexts, dtype=float):
        seen = c[np.isfinite(c)]
        series.append(seen[-int(cfg.context):] if len(seen) else np.zeros(1))
    out = _model(cfg).predict_quantiles(series, prediction_length=h,
                                        quantile_levels=list(levels))
    quant = np.asarray(out[0] if isinstance(out, tuple) else out, dtype=float)
    quant = quant.reshape(len(series), h, -1)
    mid = quant.shape[-1] // 2
    return quant[:, :, mid], quant
