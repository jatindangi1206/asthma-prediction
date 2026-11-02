"""TimesFM (Google), frozen — the forecasting benchmark row.

    python -m wearable forecast timesfm --signal hr

The sibling of `timestral.py`, identical but for the checkpoint. `_base.py` does the
work; read `timestral.py` for why a frozen forecaster is truncation-exact.

CHECKPOINT AND LICENCE — READ BEFORE CHANGING THE DEFAULT. The default is **2.5**
(`google/timesfm-2.5-200m-pytorch`): Apache-2.0, with a stable `forecast()` surface.
TimesFM **3.0** weights are under the TimesFM Non-Commercial License v1.0 — fine for
this research, not for anything shipped — and its Python API moved between releases:
published examples disagree on whether the entry point is `timesfm.ModelConfig` +
`TimesFM3Evaluator` or a `timesfm3` module, and the reference COPD repo reaches
`timesfm3.torch.model.TimesFM3Torch` directly. **The 3.0 forecast signature is
unverified here, so it is not the default and is not guessed at.** To use it: install
its package, point `repo` at the 3.0 checkpoint and adapt `forecast` below — the only
place the vendor API is touched. Check the installed signature first.

Unlike Timestral it does not document NaN as a missing marker, so each context is passed
as the readings it actually holds: the same information without the explicit spacing.
That is the one behavioural difference between the two adapters, and it is why Timestral
is preferred on a series this sparse.

    pip install "timesfm[torch]"
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wearable.forecasting import LEVELS
from wearable.forecasting._base import ForecastConfig

LABEL = "TimesFM zero-shot forecast (frozen)"
_MODELS: dict = {}


@dataclass
class TimesFMConfig(ForecastConfig):
    repo: str = "google/timesfm-2.5-200m-pytorch"   # Apache-2.0; 3.0 is non-commercial
    device: str | None = None


CONFIG = TimesFMConfig
DEFAULTS = {"repo": "google/timesfm-2.5-200m-pytorch", "context": 512, "stride": 32,
            "min_context": 64}


def _model(cfg, horizon: int):
    """Load and compile once per process, at first use — never at import time."""
    key = (cfg.repo, int(cfg.context), int(horizon))
    if key not in _MODELS:
        try:
            import timesfm
        except ImportError as exc:
            raise ImportError(
                "the `timesfm` package is needed for the timesfm forecaster: "
                'pip install "timesfm[torch]"') from exc
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(cfg.repo)
        model.compile(timesfm.ForecastConfig(
            max_context=int(cfg.context), max_horizon=int(horizon),
            normalize_inputs=True, use_continuous_quantile_head=True))
        _MODELS[key] = model
    return _MODELS[key]


def forecast(contexts, horizon, config=None, levels=LEVELS):
    """(n, T) contexts -> (median, quantiles | None). The only vendor call."""
    cfg = config or TimesFMConfig()
    h = int(horizon)
    series = []
    for c in np.asarray(contexts, dtype=float):
        seen = c[np.isfinite(c)]
        series.append(seen[-int(cfg.context):] if len(seen) else np.zeros(1))
    point, quant = _model(cfg, h).forecast(horizon=h, inputs=series)
    median = np.asarray(point, dtype=float).reshape(len(series), h)
    q = None if quant is None else np.asarray(quant, dtype=float).reshape(
        len(series), h, -1)
    return median, q
