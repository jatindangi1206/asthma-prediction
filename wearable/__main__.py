"""One entry point: `python -m wearable <command> [...]`.

Each step runs on its own; `chain` and `sweep` are the only things that connect them.

    python -m wearable list                         every discovered method
    python -m wearable selftest                     assert the invariants, touch no data
    python -m wearable runs [step] [method]         the artifact log (provenance)
    python -m wearable lineage <step> <method> <key>  trace a result back to the source

    python -m wearable ingest [--source pass1] [--channels hrv,hr]
    python -m wearable smooth gammadglm [--signal hrv|hr|temp|steps] [--deltrend 0.995]
    python -m wearable forecast timestral [--signal hr]      frozen TSFM -> surprise
    python -m wearable encode moment [--signal hr] [--days 7] [--profile 73]
    python -m wearable detect --on raw|smoothed|residual|surprise|tail
                             [--methods bocpd,kcpd] [--smoothing gammadglm]
                             [--forecasting timestral] [--signal hr]

Every channel runs the same ten smoothers and four detectors: `smooth_dataframe` takes
the value column as an argument, so nothing in a pool is HRV-specific. Each channel has
its own ingest files, so H8 segments it on its own observation times — heart rate is on
~100% of raw rows and HRV on ~6%, and a shared grid would hide one channel's gaps
behind the other's readings.
    python -m wearable join [--artifacts smooth:gammadglm,detect:bocpd] [--target detection]
                            [--days 7] [--horizon 1]
    python -m wearable model <join-key> [--model xgboost]
    python -m wearable evaluate <step> <method> <key>

    python -m wearable chain [--smoothing gammadglm] [--cpd bocpd,kcpd] [--on residual]
    python -m wearable sweep smoothing|cpd|signal|smoothers-no-model
    python -m wearable sweep inputs|models|window      the modelling experiment

The modelling experiment runs one task over the input variants and the fold grid:

    python -m wearable sweep inputs --task detection
    python -m wearable sweep inputs --task onset --variants raw,residual+cpd
    python -m wearable sweep inputs --model_names xgboost,catboost
    python -m wearable sweep models --task onset --variant residual+cpd
    python -m wearable sweep inputs --variants raw --test_pct 20 --seeds 0 --boot 0

Inputs: raw · smoothed · raw+cpd · smoothed+cpd · residual+cpd. Tasks: detection (is
today an exacerbation day?) and onset (does a new episode begin within --horizon days?).
Every variant in a run shares the task, the model, the folds and the test patients, so
the only thing that differs is the input. Tables land in data/reports/experiments/.

The window ablation holds all of that fixed and varies only how much history the model
is given — 7 days of input, then 6, 5, ... down to the classified day alone:

    python -m wearable sweep window --task detection
    python -m wearable sweep window --task onset --horizon 1
    python -m wearable sweep window --days 7,3,1 --model_names xgboost,lstm
    python -m wearable sweep window --task detection --variants raw --test_pct 20 --seeds 0

`--days` and `--horizon` belong to `join`, never to a model: the window is built as
`<feature>_d0 ... _d<N-1>` lag columns, so every model in the pool reads every window
with no change. Detection always asks about the classified day itself; `--horizon`
narrows onset to "an episode starts within this many days" (1 = the next day only).

Every step reuses a cached artifact when the method, settings, input and code version
all match. `--force` recomputes.
"""
from __future__ import annotations

import argparse
import ast
import sys

import pandas as pd

from wearable import config, cpd, metrics, models, registry, smoothing, sources, store


def _knobs(pairs: list[str]) -> dict:
    """['--deltrend','0.995'] -> {'deltrend': 0.995}; values parsed as Python literals."""
    out, it = {}, iter(pairs)
    for tok in it:
        if not tok.startswith("--"):
            raise SystemExit(f"expected --knob before {tok!r}")
        try:
            raw = next(it)
        except StopIteration:
            raise SystemExit(f"{tok} needs a value")
        k = tok[2:].replace("-", "_")
        try:
            out[k] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[k] = raw
    return out


def _show(df: pd.DataFrame) -> None:
    if df is None or not len(df):
        print("  (nothing)")
        return
    with pd.option_context("display.width", 200, "display.max_columns", 50,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print(df.to_string(index=False))


def _list() -> None:
    print("\nsources        python -m wearable ingest --source <key>")
    for k, s in sources.REGISTRY.items():
        print(f"  {k:<14} {s.label}{'' if s.has_labels else '   [no labels]'}")
    for pool, hint in (("smoothing", "python -m wearable smooth <key>"),
                       ("forecasting", "python -m wearable forecast <key> --signal hr"),
                       ("cpd", "python -m wearable detect --methods <key>"),
                       ("encoders", "python -m wearable encode <key> --signal hr"),
                       ("models", "python -m wearable model <join-key> --model <key>")):
        print(f"\n{pool:<14} {hint}")
        for k, m in registry.registry(pool).items():
            knobs = "  ".join(f"{a}={b}" for a, b in m.defaults.items())
            print(f"  {k:<14} {m.label}")
            if knobs:
                print(f"  {'':<14} {knobs}")
    print("\nmetrics        auto-applied by `evaluate`")
    for k, m in registry.registry("metrics").items():
        print(f"  {k:<14} [{m.kind}] {m.label}")
    from wearable import experiment
    from wearable.steps import join as join_step
    print("\ninputs         python -m wearable sweep inputs --variants <key>,<key>")
    for k, spec in experiment.INPUTS.items():
        on = spec["detect_on"]
        print(f"  {k:<14} {'a smoother' if spec['smooth'] else 'no smoothing'}"
              f"{f', CPD on the {on} signal' if on else ', no CPD'}")
    print(f"\ntasks          {' · '.join(sorted(join_step.TARGETS))}"
          "   (--task; label and validity columns per task)")
    print(f"\nwindow         python -m wearable sweep window --days <n>,<n> "
          f"[--horizon <n>]")
    print(f"  --days       {list(config.WINDOW_DAYS)}  the classified day + the "
          f"days-1 before it, as _d<k> lag columns (default {config.DAYS})")
    print(f"  --horizon    onset only, 1..{config.HORIZON}  an episode starts within "
          f"this many days (ablation {list(config.WINDOW_HORIZON)})")
    print(f"\nensemble       {list(cpd.DEFAULT_ENSEMBLE)}")
    print(f"store format   {store.store_format()}  (set WEARABLE_FORMAT to change)")
    print("legacy         pf (SMC+FFBS) is non-causal — legacy/run_pf_filters.py\n")


def _resolve(spec: str):
    """'<step>:<method>[:<key>]' -> the newest complete matching artifact.

    Method names legitimately contain colons (an evaluate artifact is
    `evaluate/model:logreg`), so the spec is split at the FIRST colon only and a
    trailing key is recognised by being a real artifact directory.
    """
    step, _, method = spec.partition(":")
    if not method:
        raise SystemExit(f"expected step:method[:key], got {spec!r}")

    # trailing short key? only if that directory actually exists
    head, _, tail = method.rpartition(":")
    if head and (store.artifact_dir(step, head, tail) / "manifest.json").exists():
        method, short = head, tail
        return store.load(step, method, _full_key(step, method, short))

    rows = store.ls(step, method)
    rows = rows[rows["status"] == "complete"] if len(rows) else rows
    if not len(rows):
        raise SystemExit(f"no complete artifact for {spec!r} — run that step first "
                         f"(see `python -m wearable runs`)")
    return store.load(step, method, _full_key(step, method, rows.iloc[0]["key"]))


def _full_key(step: str, method: str, short: str) -> str:
    import json
    return json.loads(
        (store.artifact_dir(step, method, short) / "manifest.json").read_text())["key"]


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        raise SystemExit(__doc__)
    cmd, rest = argv[0], argv[1:]

    if cmd == "list":
        return _list()
    if cmd == "selftest":
        from wearable import selftest
        return selftest.run()
    if cmd == "runs":
        return _show(store.ls(*(rest[:2])))
    if cmd == "lineage":
        if not rest:
            raise SystemExit("lineage needs step:method[:key] (or step method [key])")
        # accept both `lineage model:logreg` and `lineage model logreg`
        art = _resolve(":".join(p for tok in rest for p in tok.split(":")))
        for i, m in enumerate(art.lineage()):
            print(f"{'  ' * i}{m.get('step')}/{m.get('method')} "
                  f"key={str(m.get('key'))[:12]} params={m.get('params')} "
                  f"code={m.get('code', {}).get('git_rev', '')[:8]}"
                  f"{' DIRTY' if m.get('code', {}).get('git_dirty') else ''}")
        return

    if cmd == "ingest":
        from wearable.steps import ingest
        ap = argparse.ArgumentParser(prog="wearable ingest")
        ap.add_argument("--source", default=sources.DEFAULT_SOURCE,
                        choices=sorted(sources.REGISTRY))
        ap.add_argument("--workers", type=int, default=None)
        ap.add_argument("--channels", default="",
                        help="comma-separated channels to write; "
                             f"default all of {','.join(config.CHANNELS)}")
        a = ap.parse_args(rest)
        return ingest.run(a.source, a.workers,
                          channels=[c for c in a.channels.split(",") if c] or None)

    if cmd == "smooth":
        from wearable.steps import smooth
        if not rest:
            raise SystemExit(f"smooth needs a method: {smoothing.names()}")
        force = "--force" in rest
        rest = [r for r in rest if r != "--force"]
        smooth.run(rest[0], force=force, **_knobs(rest[1:]))
        return

    if cmd == "detect":
        from wearable.steps import detect, smooth
        ap = argparse.ArgumentParser(prog="wearable detect")
        ap.add_argument("--on", default="residual", choices=sorted(detect.SIGNALS))
        ap.add_argument("--methods", default=",".join(cpd.DEFAULT_ENSEMBLE))
        ap.add_argument("--smoothing", default="gammadglm")
        ap.add_argument("--forecasting", default="timestral",
                        help="which forecaster, for --on surprise|tail")
        ap.add_argument("--signal", default=config.DEFAULT_CHANNEL,
                        choices=sorted(config.CHANNELS),
                        help="which channel to detect on")
        ap.add_argument("--force", action="store_true")
        a = ap.parse_args(rest)
        if a.on == "raw":
            src = None
        elif a.on in ("surprise", "tail"):
            from wearable.steps import forecast as fc
            src = fc.run(a.forecasting, force=a.force, signal=a.signal)
        else:
            src = smooth.run(a.smoothing, force=a.force, signal=a.signal)
        detect.run(a.methods.split(","), on=a.on, input_artifact=src,
                   force=a.force, signal=a.signal)
        return

    if cmd == "forecast":
        from wearable.steps import forecast as fc
        if not rest:
            from wearable import forecasting
            raise SystemExit(f"forecast needs a method: {forecasting.names()}")
        force = "--force" in rest
        rest = [r for r in rest if r != "--force"]
        fc.run(rest[0], force=force, **_knobs(rest[1:]))
        return

    if cmd == "encode":
        from wearable.steps import encode as en
        if not rest:
            from wearable import encoders
            raise SystemExit(f"encode needs a method: {encoders.names()}")
        force = "--force" in rest
        rest = [r for r in rest if r != "--force"]
        en.run(rest[0], force=force, **_knobs(rest[1:]))
        return

    if cmd == "join":
        from wearable.steps import join
        ap = argparse.ArgumentParser(prog="wearable join")
        ap.add_argument("--artifacts", default="",
                        help="comma-separated step:method[:key], empty = raw baseline")
        ap.add_argument("--target", default="detection", choices=sorted(join.TARGETS))
        ap.add_argument("--days", type=int, default=config.DAYS,
                        help="input window: the classified day + the days-1 before it")
        ap.add_argument("--horizon", type=int, default=config.HORIZON,
                        help=f"onset only: a new episode within this many days "
                             f"(1..{config.HORIZON})")
        ap.add_argument("--cohort", default="all", choices=list(join.ANALYSIS_SETS),
                        help="analysis population; 'all' is the primary design")
        ap.add_argument("--patients", default="",
                        help="comma-separated ids, overriding --cohort")
        ap.add_argument("--channels", default=",".join(config.DEFAULT_JOIN_CHANNELS),
                        help="comma-separated channels the raw baseline reads, e.g. "
                             "hrv,hr,sleep_active")
        ap.add_argument("--profile", type=int, default=config.PROFILE,
                        help=f"time-of-day slots of raw signal per day, 0 = none; "
                             f"{config.PROFILE_BINS} = the native 10-min cadence. "
                             f"Needed by the foundation models")
        ap.add_argument("--force", action="store_true")
        a = ap.parse_args(rest)
        arts = [_resolve(s) for s in a.artifacts.split(",") if s]
        join.run(arts, target=a.target, force=a.force,
                 days=a.days, horizon=a.horizon, profile=a.profile,
                 channels=[c for c in a.channels.split(",") if c],
                 patients=([p for p in a.patients.split(",") if p]
                           or (None if a.cohort == "all"
                               else join.analysis_set(a.cohort))))
        return

    if cmd == "model":
        from wearable.steps import model as ms
        ap = argparse.ArgumentParser(prog="wearable model")
        ap.add_argument("join_spec", help="join:<name>[:key]")
        ap.add_argument("--model", default=models.DEFAULT_MODEL, choices=models.names())
        ap.add_argument("--force", action="store_true")
        a = ap.parse_args(rest)
        ms.run(_resolve(a.join_spec), model=a.model, force=a.force)
        return

    if cmd == "evaluate":
        from wearable.steps import evaluate
        if not rest:
            raise SystemExit("evaluate needs step:method[:key]")
        art = evaluate.run(_resolve(rest[0]), force="--force" in rest)
        return _show(art.read_table("metrics"))

    if cmd == "chain":
        from wearable import experiment
        ap = argparse.ArgumentParser(prog="wearable chain")
        ap.add_argument("--smoothing", default=None)
        ap.add_argument("--cpd", default=None)
        ap.add_argument("--on", default="residual")
        ap.add_argument("--model", default=models.DEFAULT_MODEL)
        ap.add_argument("--target", default="detection")
        ap.add_argument("--days", type=int, default=config.DAYS)
        ap.add_argument("--horizon", type=int, default=config.HORIZON)
        ap.add_argument("--profile", type=int, default=config.PROFILE)
        ap.add_argument("--force", action="store_true")
        a = ap.parse_args(rest)
        r = experiment.chain(smoothing_method=a.smoothing,
                             cpd_methods=a.cpd.split(",") if a.cpd else None,
                             on=a.on, model=a.model, target=a.target, force=a.force,
                             days=a.days, horizon=a.horizon, profile=a.profile)
        return _show(r["evaluate"].read_table("metrics"))

    if cmd == "sweep":
        from wearable import experiment
        which = rest[0] if rest else "smoothing"
        opts = _knobs(rest[1:])
        # `--variants raw,smoothed` is friendlier than quoting a Python list, and
        # `--test_pct 20,30` already parses as a tuple literal.
        opts = {k: (v.split(",") if isinstance(v, str) and "," in v else v)
                for k, v in opts.items()}
        fn = {"smoothing": experiment.sweep_smoothing,
              "cpd": experiment.sweep_cpd,
              "signal": experiment.sweep_signal,
              "smoothers-no-model": experiment.compare_smoothers_no_model,
              "inputs": experiment.sweep_inputs,
              "models": experiment.sweep_models,
              "window": experiment.sweep_window}.get(which)
        if fn is None:
            raise SystemExit("sweep takes: smoothing | cpd | signal | "
                             "smoothers-no-model | inputs | models | window")
        print(f"\n{'=' * 70}\nCOMPARISON: {which}\n{'=' * 70}")
        return _show(fn(**opts))

    raise SystemExit(f"unknown command {cmd!r}\n{__doc__}")


if __name__ == "__main__":
    main()
