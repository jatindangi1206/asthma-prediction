"""The experiment runner — the ONLY place that connects the steps.

Every step is independent and knows nothing about the others. This module wires them
into a chain, and sweeps one axis at a time while holding everything else fixed:

    sweep_smoothing()   every smoothing method + a no-smoothing baseline,
                        each through the SAME model, seed and test participants
    sweep_cpd()         every CPD method, on the same input signal, same model
    sweep_signal()      one CPD ensemble on raw vs smoothed vs residual
    chain()             one explicit pipeline, any subset of steps, in any order

What is held constant in a sweep: the target, the seed, the model, the test
participants (the split comes from the patient universe and the seed alone, never from
the features), and every step's cached artifacts. So a difference between two rows is
the method under test and nothing else.

Because every step caches on a content key, re-running a sweep after adding one method
recomputes only that method.
"""
from __future__ import annotations

import pandas as pd

from wearable import cpd, metrics, models, registry, smoothing
from wearable.config import SEED, TEST_FRACTION
from wearable.steps import detect, evaluate, join, model as model_step, smooth

NO_SMOOTHING = "none"


def chain(smoothing_method: str | None = None, cpd_methods=None,
          on: str = "residual", model: str = models.DEFAULT_MODEL,
          target: str = "detection", seed: int = SEED,
          test_fraction: float = TEST_FRACTION, force: bool = False,
          smoothing_params: dict | None = None, limit: int | None = None):
    """Run one pipeline. Every argument is optional — omit smoothing and CPD and the
    raw signal goes straight to the model."""
    artifacts = []
    sm = None
    if smoothing_method and smoothing_method != NO_SMOOTHING:
        sm = smooth.run(smoothing_method, force=force, limit=limit,
                        **(smoothing_params or {}))
        artifacts.append(sm)

    if cpd_methods:
        if on != "raw" and sm is None:
            raise ValueError(f"on={on!r} needs a smoothing method; "
                             f"use on='raw' to detect on the unsmoothed signal")
        artifacts.append(detect.run(cpd_methods, on=on,
                                    input_artifact=sm if on != "raw" else None,
                                    force=force, limit=limit))

    j = join.run(artifacts, target=target, seed=seed,
                 test_fraction=test_fraction, force=force)
    m = model_step.run(j, model=model, seed=seed, force=force)
    e = evaluate.run(m, force=force)
    return {"smooth": sm, "join": j, "model": m, "evaluate": e,
            "artifacts": artifacts}


def _row(variant: str, result: dict) -> dict:
    stats = result["evaluate"].manifest["stats"]
    mstats = result["model"].manifest["stats"]
    row = {"variant": variant,
           "n_train": mstats.get("n_train"), "n_test": mstats.get("n_test"),
           "test_positives": mstats.get("test_positives"),
           "n_features": mstats.get("n_features")}
    row.update({k: v for k, v in stats.items() if k in metrics.names("model")})
    row["model_key"] = result["model"].key[:12]
    return row


def _table(rows: list[dict], sort_by: str | None = None) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    key = sort_by or next((m for m in ("auprc", "auroc") if m in df.columns), None)
    if key and key in df.columns:
        df = df.sort_values(key, ascending=not metrics.higher_is_better(key))
    return df.reset_index(drop=True)


def sweep_smoothing(methods=None, include_baseline: bool = True,
                    model: str = models.DEFAULT_MODEL, target: str = "detection",
                    seed: int = SEED, force: bool = False, sort_by: str | None = None,
                    limit: int | None = None):
    """Every smoothing method through the same model. The baseline row is no smoothing
    at all, which is the comparison that says whether smoothing helps."""
    methods = list(methods or smoothing.names())
    variants = ([NO_SMOOTHING] if include_baseline else []) + methods
    rows = []
    for v in variants:
        print(f"\n{'=' * 70}\nvariant: smoothing = {v}\n{'=' * 70}")
        try:
            rows.append(_row(v, chain(smoothing_method=v, model=model, target=target,
                                      seed=seed, force=force, limit=limit)))
        except Exception as exc:
            print(f"  ! variant {v} failed: {type(exc).__name__}: {exc}")
            rows.append({"variant": v, "error": f"{type(exc).__name__}: {exc}"})
    return _table(rows, sort_by)


def sweep_cpd(methods=None, on: str = "residual", smoothing_method: str = "gammadglm",
              model: str = models.DEFAULT_MODEL, target: str = "detection",
              seed: int = SEED, force: bool = False, sort_by: str | None = None,
              limit: int | None = None):
    """Every CPD method on the same input signal, through the same model."""
    methods = list(methods or cpd.names())
    rows = []
    for v in methods:
        print(f"\n{'=' * 70}\nvariant: cpd = {v} on {on}\n{'=' * 70}")
        try:
            rows.append(_row(v, chain(
                smoothing_method=None if on == "raw" else smoothing_method,
                cpd_methods=[v], on=on, model=model, target=target,
                seed=seed, force=force, limit=limit)))
        except Exception as exc:
            print(f"  ! variant {v} failed: {type(exc).__name__}: {exc}")
            rows.append({"variant": v, "error": f"{type(exc).__name__}: {exc}"})
    return _table(rows, sort_by)


def sweep_signal(cpd_methods=cpd.DEFAULT_ENSEMBLE, smoothing_method: str = "gammadglm",
                 signals=("raw", "smoothed", "residual"),
                 model: str = models.DEFAULT_MODEL, target: str = "detection",
                 seed: int = SEED, force: bool = False, sort_by: str | None = None,
              limit: int | None = None):
    """One CPD ensemble, three input signals — does de-trending actually help detection?"""
    rows = []
    for on in signals:
        print(f"\n{'=' * 70}\nvariant: detect on {on}\n{'=' * 70}")
        try:
            rows.append(_row(on, chain(
                smoothing_method=None if on == "raw" else smoothing_method,
                cpd_methods=list(cpd_methods), on=on, model=model,
                target=target, seed=seed, force=force, limit=limit)))
        except Exception as exc:
            print(f"  ! variant {on} failed: {type(exc).__name__}: {exc}")
            rows.append({"variant": on, "error": f"{type(exc).__name__}: {exc}"})
    return _table(rows, sort_by)


def compare_smoothers_no_model(methods=None, force: bool = False,
                               limit: int | None = None):
    """Judge smoothing methods without any model: raw vs smoothed, signal metrics only."""
    rows = []
    for m in list(methods or smoothing.names()):
        try:
            art = evaluate.run(smooth.run(m, force=force, limit=limit), force=force)
            row = {"variant": m}
            row.update({r["metric"]: r["value"]
                        for _, r in art.read_table("metrics").iterrows()})
            rows.append(row)
        except Exception as exc:
            print(f"  ! {m} failed: {type(exc).__name__}: {exc}")
            rows.append({"variant": m, "error": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows).reset_index(drop=True)
