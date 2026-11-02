"""The experiment runner — the ONLY place that connects the steps.

Every step is independent and knows nothing about the others. This module wires them
into a chain, and sweeps one axis at a time while holding everything else fixed:

    sweep_smoothing()   every smoothing method + a no-smoothing baseline,
                        each through the SAME model, seed and test participants
    sweep_cpd()         every CPD method, on the same input signal, same model
    sweep_signal()      one CPD ensemble on raw vs smoothed vs residual
    sweep_inputs()      the modelling experiment: every INPUT variant (raw, smoothed,
                        raw+cpd, smoothed+cpd, residual+cpd) over the full fold grid
    sweep_models()      the same grid with the model as the axis instead
    sweep_window()      the same grid with the INPUT WINDOW as the axis: 7 days of
                        history, then 6, ... down to the classified day alone (and for
                        onset, how far ahead the episode is asked about)
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
from wearable.config import (DAYS, DEFAULT_CHANNEL, DEFAULT_JOIN_CHANNELS,
                             HORIZON, PROFILE, PROFILE_BINS,
                             REPORTS_DIR, SEED, SEEDS, TEST_FRACTION, TEST_PCT,
                             WINDOW_DAYS, WINDOW_HORIZON)
from wearable.steps import detect, evaluate, join, model as model_step, smooth

NO_SMOOTHING = "none"


def chain(smoothing_method: str | None = None, cpd_methods=None,
          on: str = "residual", model: str = models.DEFAULT_MODEL,
          target: str = "detection", seed: int = SEED,
          test_fraction: float = TEST_FRACTION, force: bool = False,
          smoothing_params: dict | None = None, limit: int | None = None,
          days: int = DAYS, horizon: int = HORIZON, profile: int = PROFILE):
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
                 test_fraction=test_fraction, force=force,
                 days=days, horizon=horizon, profile=profile)
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
    key = sort_by or next((m for m in ("auroc", "sensitivity") if m in df.columns), None)
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


# ===========================================================================
# The modelling experiment: two tasks, interchangeable inputs, interchangeable
# models, one fold grid.
# ===========================================================================
# An INPUT variant says what the model is allowed to see. Each is a recipe for the
# artifact list `join` receives — nothing more — so adding a variant is one dict entry
# and no new machinery. The signal columns come along for free: a smooth artifact's
# table carries hrvValue, smoothed_hrv and residual_hrv, and a detect artifact's table
# carries the signal it ran on beside its anomaly flags, so "raw+cpd" really does give
# the model the raw day features as well as the change-point features.
#
#   raw            no smoothing, no CPD — the day features of the ingested signal
#   smoothed       a smoother's output (and its residual) with no CPD
#   raw+cpd        CPD run straight on the raw signal
#   smoothed+cpd   a smoother, plus CPD on the smoothed signal
#   residual+cpd   a smoother, plus CPD on raw - smoothed (the de-trended track)
#
# Any other legal pair also works: inputs(smooth_method=..., detect_on=...).
DEFAULT_FORECASTER = "timestral"
DEFAULT_ENCODER = "moment"

# A variant is a dict of four switches and nothing else, so adding one is one entry.
#   smooth      run a smoother and hand join its artifact
#   detect_on   which column CPD runs on, or None for no CPD
#   forecast     run a frozen forecaster (predicted / surprise / tail), or None
#   encode       run a frozen encoder (an embedding per patient-day), or None
INPUTS: dict[str, dict] = {
    "raw":          {"smooth": False, "detect_on": None, "forecast": None, "encode": None},
    "smoothed":     {"smooth": True, "detect_on": None, "forecast": None, "encode": None},
    "raw+cpd":      {"smooth": False, "detect_on": "raw", "forecast": None, "encode": None},
    "smoothed+cpd": {"smooth": True, "detect_on": "smoothed", "forecast": None, "encode": None},
    "residual+cpd": {"smooth": True, "detect_on": "residual", "forecast": None, "encode": None},
    # frozen foundation routes
    "forecast":     {"smooth": False, "detect_on": None, "forecast": DEFAULT_FORECASTER, "encode": None},
    "surprise+cpd": {"smooth": False, "detect_on": "surprise", "forecast": DEFAULT_FORECASTER, "encode": None},
    "embed":        {"smooth": False, "detect_on": None, "forecast": None, "encode": DEFAULT_ENCODER},
    "embed+raw":    {"smooth": False, "detect_on": None, "forecast": None, "encode": DEFAULT_ENCODER},
}
DEFAULT_SMOOTHING = "gammadglm"
EXPERIMENT_DIR = REPORTS_DIR / "experiments"


def _as_list(value, default=None):
    """A scalar, a comma-separated string or any sequence -> a list.

    `--test_pct 20` and `--variants raw,smoothed` both reach here from the CLI, and a
    bare string must never be iterated character by character.
    """
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        return [s for s in (v.strip() for v in value.split(",")) if s]
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


def inputs(smooth_method: str | None = None, detect_on: str | None = None,
           cpd_methods=cpd.DEFAULT_ENSEMBLE, force: bool = False,
           limit: int | None = None, smoothing_params: dict | None = None,
           forecast_method: str | None = None, encode_method: str | None = None,
           signal: str = DEFAULT_CHANNEL, days: int = DAYS,
           profile: int = PROFILE_BINS):
    """The artifacts `join` should receive for one input variant, in join order.

    Each of smoothing, forecasting and encoding is an independent frozen stage, and CPD
    runs on whichever of their columns `detect_on` names — `raw`/`smoothed`/`residual`
    come from a smoother, `surprise`/`tail` from a forecaster.
    """
    from wearable.steps import encode as encode_step, forecast as forecast_step
    artifacts = []
    sm = fc = None
    if smooth_method and smooth_method != NO_SMOOTHING:
        sm = smooth.run(smooth_method, force=force, limit=limit, signal=signal,
                        **(smoothing_params or {}))
        artifacts.append(sm)
    if forecast_method:
        fc = forecast_step.run(forecast_method, signal=signal, force=force, limit=limit)
        artifacts.append(fc)
    if detect_on:
        src = {"raw": None, "smoothed": sm, "residual": sm,
               "surprise": fc, "tail": fc}[detect_on]
        if detect_on != "raw" and src is None:
            need = "forecaster" if detect_on in ("surprise", "tail") else "smoother"
            raise ValueError(f"detect_on={detect_on!r} needs a {need}; "
                             f"use 'raw' to detect on the unsmoothed signal")
        artifacts.append(detect.run(cpd_methods, on=detect_on, input_artifact=src,
                                    force=force, limit=limit, signal=signal))
    if encode_method:
        artifacts.append(encode_step.run(encode_method, signal=signal, days=days,
                                         profile=profile, force=force))
    return artifacts


def variant_inputs(variant: str, smoothing_method: str = DEFAULT_SMOOTHING, **kw):
    """The artifact list of a named INPUT variant."""
    if variant not in INPUTS:
        raise ValueError(f"unknown input variant {variant!r}; "
                         f"available: {sorted(INPUTS)}")
    spec = INPUTS[variant]
    return inputs(smoothing_method if spec["smooth"] else None,
                  spec["detect_on"], forecast_method=spec.get("forecast"),
                  encode_method=spec.get("encode"), **kw)


def run_one(variant: str = "raw", task: str = "detection",
            model: str = models.DEFAULT_MODEL,
            smoothing_method: str = DEFAULT_SMOOTHING,
            cpd_methods=cpd.DEFAULT_ENSEMBLE, test_fraction: float = TEST_FRACTION,
            seed: int = SEED, force: bool = False, limit: int | None = None,
            boot: int | None = None, stratify: bool = True,
            smoothing_params: dict | None = None, model_params: dict | None = None,
            days: int = DAYS, horizon: int = HORIZON, profile: int = PROFILE,
            channels=DEFAULT_JOIN_CHANNELS, patients=None):
    """One cell of the experiment: one input variant, one task, one model, one fold.

    The seed drives both the split and the model, as in the reference design, so a cell
    is identified by (variant, task, model, days, horizon, test_fraction, seed) and
    nothing else. `days` and `horizon` go to `join`, never to the model: the window is a
    property of the dataset, so every model in the pool reads every window unchanged.
    """
    arts = variant_inputs(variant, smoothing_method, cpd_methods=cpd_methods,
                          force=force, limit=limit, smoothing_params=smoothing_params)
    j = join.run(arts, target=task, seed=seed, test_fraction=test_fraction,
                 force=force, stratify=stratify, days=days, horizon=horizon,
                 profile=profile, channels=channels, patients=patients)
    m = model_step.run(j, model=model, seed=seed, force=force,
                       **(model_params or {}))
    e = evaluate.run(m, force=force, boot=boot)
    return {"variant": variant, "task": task, "model": model, "join": j,
            "model_artifact": m, "evaluate": e, "artifacts": arts}


def _metric_row(art) -> dict:
    """An evaluate artifact's metrics table flattened to one row, with the intervals."""
    out = {}
    for _, r in art.read_table("metrics").iterrows():
        out[r["metric"]] = r["value"]
        for b in ("lo", "hi"):
            if b in r.index and pd.notna(r[b]):
                out[f"{r['metric']}_{b}"] = r[b]
    return out


def _cell_rows(res: dict, test_pct: int, seed: int) -> list[dict]:
    """One row for all test patients, plus one per cohort group."""
    js, ms = res["join"].manifest["stats"], res["model_artifact"].manifest["stats"]
    head = {"task": res["task"], "input": res["variant"], "model": res["model"],
            "days": js.get("days"), "horizon": js.get("horizon"),
            "profile": js.get("profile"),
            "test_pct": test_pct, "seed": seed,
            "n_features": ms.get("n_features"), "n_train": ms.get("n_train"),
            "n_test": ms.get("n_test"), "test_positives": js.get("test_positives"),
            "test_events": js.get("test_events"),
            "n_test_patients": js.get("n_test_patients"),
            "model_key": res["model_artifact"].key[:12]}
    rows = [{**head, "group": "all", **_metric_row(res["evaluate"])}]
    if res["evaluate"].has_table("metrics_by_group"):
        g = res["evaluate"].read_table("metrics_by_group")
        for name, part in g.groupby("group", sort=True):
            rows.append({**head, "group": str(name),
                         "n_test": int(part["n_days"].iloc[0]),
                         "test_positives": int(part["n_pos_days"].iloc[0]),
                         "test_events": int(part["n_events"].iloc[0])
                         if "n_events" in part.columns else None,
                         "n_test_patients": int(part["n_patients"].iloc[0])
                         if "n_patients" in part.columns else None,
                         **dict(zip(part["metric"], part["value"]))})
    return rows


def _cohort(cohort):
    """A named analysis set -> its patient ids, or None for the whole labelled cohort.

    `all` is the primary design: keep the population fixed and change only the method.
    `smoothable` / `armable` are robustness checks; `exacerbators` selects on the
    outcome and is a clearly-labelled secondary analysis.
    """
    if cohort in (None, "all"):
        return None
    ids = join.analysis_set(cohort)
    print(f"  cohort  {cohort}: {len(ids)} patients")
    return ids


def _grid(variants, model_names, task, test_pcts, seeds,
          days_list=(DAYS,), horizons=(HORIZON,), cohort=None, **kw):
    """Every (variant, model, days, horizon, test share, seed) cell of one task.

    Asserts the comparison is fair: within a fold, every cell must have been scored on
    the same test days and the same test patients. That holds because the split is a
    function of (universe, seed, share, strata) alone and the universe comes from the
    labels, not from whichever method produced features — a variant that drops patients
    leaves NaN features, not missing rows.

    The fold is keyed on the horizon but NOT on the window: a narrower horizon is a
    different question and legitimately has fewer positives, while a wider window only
    adds columns. So the same assertion that keeps the inputs honest also proves that
    going from 1 to 7 days of input did not quietly change what was being scored.
    """
    patients = _cohort(cohort)
    rows, fold_shape, failures = [], {}, []
    for test_pct in test_pcts:
        for seed in seeds:
            for horizon in horizons:
                for n_days in days_list:
                    for variant in variants:
                        for name in model_names:
                            tag = (f"{task} | {variant} | {name} | {n_days}d in"
                                   + (f" | onset <= {horizon}d" if task == "onset" else "")
                                   + f" | test {test_pct}% | seed {seed}")
                            print(f"\n{'=' * 78}\n{tag}\n{'=' * 78}")
                            try:
                                res = run_one(variant, task=task, model=name,
                                              test_fraction=test_pct / 100, seed=seed,
                                              days=n_days, horizon=horizon,
                                              patients=patients, **kw)
                            except Exception as exc:
                                print(f"  ! failed: {type(exc).__name__}: {exc}")
                                failures.append({
                                    "task": task, "input": variant, "model": name,
                                    "days": n_days, "horizon": horizon,
                                    "test_pct": test_pct, "seed": seed,
                                    "error": f"{type(exc).__name__}: {exc}"})
                                continue
                            cells = _cell_rows(res, test_pct, seed)
                            shape = (cells[0]["n_test"], cells[0]["test_positives"],
                                     cells[0]["test_events"],
                                     cells[0]["n_test_patients"])
                            seen = fold_shape.setdefault(
                                (test_pct, seed, horizon), (shape, tag))
                            assert seen[0] == shape, (
                                f"unfair comparison: {tag} was scored on {shape} "
                                f"(days, positives, episodes, patients) but "
                                f"{seen[1]} on {seen[0]}")
                            rows += cells
                            top = cells[0]
                            print(f"  AUROC {top.get('auroc', float('nan')):.3f}  "
                                  f"sens {top.get('sensitivity', float('nan')):.3f}  "
                                  f"spec {top.get('specificity', float('nan')):.3f}  "
                                  f"PPV {top.get('ppv', float('nan')):.3f}  "
                                  f"Brier {top.get('brier', float('nan')):.4f}")
    return pd.DataFrame(rows), pd.DataFrame(failures)


def summarise(runs: pd.DataFrame,
              by=("task", "input", "model", "days", "horizon", "profile",
                  "test_pct", "group")):
    """Mean and SD over seeds of every score, plus the mean interval, as in the design."""
    if not len(runs):
        return runs
    by = [c for c in by if c in runs.columns]
    head = {"task", "input", "model", "days", "horizon", "profile", "test_pct",
            "seed", "group", "model_key"}
    scores = [c for c in runs.columns
              if c not in head and not c.endswith(("_lo", "_hi"))]
    agg = {"seeds": ("seed", "nunique")}
    for k in scores:
        agg[f"{k}_mean"], agg[f"{k}_sd"] = (k, "mean"), (k, "std")
    for k in scores:
        for b in ("lo", "hi"):
            if f"{k}_{b}" in runs.columns:
                agg[f"{k}_{b}"] = (f"{k}_{b}", "mean")
    return runs.groupby(list(by), dropna=False).agg(**agg).reset_index()


CELL = ["task", "input", "model", "days", "horizon", "profile", "test_pct", "seed",
        "group"]


def _read_runs(runs_path):
    """Whatever is already recorded for this task, or None.

    A file that cannot be parsed is moved aside rather than dropped or allowed to block
    every later run.
    """
    if not runs_path.exists():
        return None
    try:
        return pd.read_csv(runs_path)
    except Exception as exc:
        spoiled = runs_path.with_suffix(".unreadable.csv")
        runs_path.rename(spoiled)
        print(f"  ! {runs_path} did not parse ({type(exc).__name__}); "
              f"kept as {spoiled} and starting a new one")
        return None


def _report(runs: pd.DataFrame, failures: pd.DataFrame, task: str, axis: str):
    """Record this run, then summarise every run of the task recorded so far.

    The file is read, merged and REWRITTEN rather than appended to: a run with other
    settings carries a different column set — `--boot 0` drops every `_lo`/`_hi` — and
    appending those rows under an older header corrupts the file. A re-run of the same
    cell replaces the earlier one, so a fixed metric or a changed method cannot leave a
    stale row shadowing the table.
    """
    if not len(runs):
        print("  (no run completed)")
        return runs
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    runs_path = EXPERIMENT_DIR / f"runs_{task}.csv"
    earlier = _read_runs(runs_path)
    recorded = (pd.concat([earlier, runs], ignore_index=True)
                if earlier is not None else runs)
    keys = [c for c in CELL if c in recorded.columns]
    recorded = recorded.drop_duplicates(keys, keep="last")
    recorded.to_csv(runs_path, index=False)
    summary = summarise(recorded)
    summary.to_csv(EXPERIMENT_DIR / f"summary_{task}.csv", index=False,
                   float_format="%.5g")
    if len(failures):
        failures.to_csv(EXPERIMENT_DIR / f"failures_{task}.csv", index=False)
    overall = summary[summary["group"] == "all"]
    for k in ("auroc", "sensitivity", "specificity", "ppv", "brier",
              "lead_days", "detection_delay"):
        if f"{k}_mean" not in overall.columns:
            continue
        print(f"\n{task} — {k}, all groups, mean over seeds")
        index = [c for c in _as_list(axis) if c in overall.columns]
        print(overall.pivot_table(index=index, columns="test_pct",
                                  values=f"{k}_mean").round(3).to_string())
    print(f"\n-> {runs_path}\n-> {EXPERIMENT_DIR / f'summary_{task}.csv'}")
    return summary


def sweep_inputs(variants=None, task: str = "detection", model_names=None,
                 test_pct=TEST_PCT, seeds=SEEDS, days=DAYS, horizon=HORIZON,
                 cohort=None, **kw):
    """The experiment with the INPUT as the axis: does smoothing or CPD earn its place?

    Every variant goes through the same model, the same folds and the same test
    patients, so a difference between two rows is the input and nothing else.
    """
    runs, failures = _grid(_as_list(variants, INPUTS),
                           _as_list(model_names, [models.DEFAULT_MODEL]), task,
                           _as_list(test_pct), _as_list(seeds),
                           days_list=_as_list(days), horizons=_as_list(horizon),
                           cohort=cohort, **kw)
    return _report(runs, failures, task, "input")


def sweep_models(model_names=None, task: str = "detection", variant: str = "raw",
                 test_pct=TEST_PCT, seeds=SEEDS, days=DAYS, horizon=HORIZON,
                 cohort=None, **kw):
    """The same experiment with the MODEL as the axis, on one fixed input."""
    runs, failures = _grid(_as_list(variant), _as_list(model_names, models.names()),
                           task, _as_list(test_pct), _as_list(seeds),
                           days_list=_as_list(days), horizons=_as_list(horizon),
                           cohort=cohort, **kw)
    return _report(runs, failures, task, "model")


def sweep_window(task: str = "detection", days=WINDOW_DAYS, horizon=None,
                 variants="raw", model_names=None, test_pct=TEST_PCT, seeds=SEEDS,
                 cohort=None, **kw):
    """The window ablation: the SAME inputs, models, folds, split and scores — only how
    much history the model is given changes.

        detection   7 days of input, then 6, 5, ... down to the classified day alone.
                    The question never moves: is THIS day an exacerbation day?
        onset       the same 7 ... 1 days of input, asking a different question:
                    does a new episode start within `horizon` days? The default is the
                    strictest one, the next day only (`config.WINDOW_HORIZON`).

    Nothing about a model changes between rows — the window is built by `join`, so the
    whole pool reads every window. A model that needs the window as a sequence rather
    than as a flat row (`rnn`, `lstm`) recovers it from the `_d<k>` column names.
    """
    horizons = _as_list(WINDOW_HORIZON if horizon is None and task == "onset"
                        else HORIZON if horizon is None else horizon)
    runs, failures = _grid(_as_list(variants), _as_list(model_names, models.names()),
                           task, _as_list(test_pct), _as_list(seeds),
                           days_list=[int(d) for d in _as_list(days)],
                           horizons=[int(h) for h in horizons], cohort=cohort,
                           **kw)
    axis = ["days", "horizon"] if task == "onset" else ["days"]
    return _report(runs, failures, task, axis)
