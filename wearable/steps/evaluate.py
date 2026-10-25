"""Step: evaluate. Scores an artifact with every applicable discovered metric.

    model predictions -> every KIND="model" metric   (y_true vs y_prob)
    a smooth artifact -> every KIND="signal" metric   (raw vs smoothed, no model needed)

    output  table `metrics`: metric, label, value, higher_is_better

Depends on no other step: it is handed an artifact and works out which metrics apply.
Adding a metric is one file in `wearable/metrics/` — it then appears here and in every
comparison table with no change to this file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wearable import metrics as metric_pool
from wearable import store

STEP = "evaluate"


def _rows(kind: str, call) -> list[dict]:
    out = []
    for name, spec in metric_pool.all_metrics(kind).items():
        if spec.label.startswith("<unavailable"):
            continue
        mod = spec.load()
        try:
            value = float(call(mod.compute))
        except Exception as exc:
            value = float("nan")
            print(f"  ! metric {name} failed: {type(exc).__name__}: {exc}")
        out.append({"metric": name, "kind": kind,
                    "label": getattr(mod, "LABEL", name), "value": value,
                    "higher_is_better": bool(getattr(mod, "HIGHER_IS_BETTER", True))})
    return out


def run(artifact, force: bool = False):
    """Pick the metric kind from what the artifact actually holds."""
    if artifact.has_table("predictions"):
        kind, table = "model", artifact.read_table("predictions")
    elif artifact.has_table("smoothed"):
        kind, table = "signal", artifact.read_table("smoothed")
    else:
        raise ValueError(f"nothing to evaluate in {artifact.step}/{artifact.method}: "
                         f"tables = {artifact.manifest.get('tables')}")

    name = f"{artifact.step}:{artifact.method}"
    params = {"kind": kind, "metrics": metric_pool.names(kind)}
    inputs = [artifact.ref]
    code_hash = "-".join(sorted(
        m.source_hash[:8] for m in metric_pool.all_metrics(kind).values()))

    if not force:
        hit = store.find(STEP, name, params, inputs, code_hash)
        if hit is not None:
            return hit

    if kind == "model":
        y, p = table["y"].to_numpy(), table["y_prob"].to_numpy()
        rows = _rows("model", lambda f: f(y, p))
    else:
        # Judge the method directly: raw against smoothed, per patient, then averaged
        # so one long-recording patient cannot dominate.
        per = []
        for _, g in table.groupby("p_id", sort=True):
            per.append({r["metric"]: r["value"] for r in
                        _rows("signal", lambda f, g=g: f(g["hrvValue"], g["smoothed_hrv"]))})
        mean = pd.DataFrame(per).mean(numeric_only=True)
        rows = _rows("signal", lambda f: np.nan)      # shape + metadata
        for r in rows:
            r["value"] = float(mean.get(r["metric"], np.nan))

    art = store.open_artifact(STEP, name, params, inputs, code_hash,
                              label=f"{kind} metrics for {name}")
    out = pd.DataFrame(rows)
    art.write_table("metrics", out)
    art.complete(kind=kind, n_metrics=len(out),
                 **{r["metric"]: r["value"] for r in rows})
    return art
