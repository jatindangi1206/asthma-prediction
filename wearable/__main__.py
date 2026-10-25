"""One entry point: `python -m wearable <command> [...]`.

Each step runs on its own; `chain` and `sweep` are the only things that connect them.

    python -m wearable list                         every discovered method
    python -m wearable selftest                     assert the invariants, touch no data
    python -m wearable runs [step] [method]         the artifact log (provenance)
    python -m wearable lineage <step> <method> <key>  trace a result back to the source

    python -m wearable ingest [--source pass1]
    python -m wearable smooth gammadglm [--deltrend 0.995]
    python -m wearable detect --on raw|smoothed|residual [--methods bocpd,kcpd]
                             [--smoothing gammadglm]
    python -m wearable join [--artifacts smooth:gammadglm,detect:bocpd] [--target detection]
    python -m wearable model <join-key> [--model logreg]
    python -m wearable evaluate <step> <method> <key>

    python -m wearable chain [--smoothing gammadglm] [--cpd bocpd,kcpd] [--on residual]
    python -m wearable sweep smoothing|cpd|signal|smoothers-no-model

Every step reuses a cached artifact when the method, settings, input and code version
all match. `--force` recomputes.
"""
from __future__ import annotations

import argparse
import ast
import sys

import pandas as pd

from wearable import cpd, metrics, models, registry, smoothing, sources, store


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
                       ("cpd", "python -m wearable detect --methods <key>"),
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
        a = ap.parse_args(rest)
        return ingest.run(a.source, a.workers)

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
        ap.add_argument("--force", action="store_true")
        a = ap.parse_args(rest)
        src = None if a.on == "raw" else smooth.run(a.smoothing, force=a.force)
        detect.run(a.methods.split(","), on=a.on, input_artifact=src, force=a.force)
        return

    if cmd == "join":
        from wearable.steps import join
        ap = argparse.ArgumentParser(prog="wearable join")
        ap.add_argument("--artifacts", default="",
                        help="comma-separated step:method[:key], empty = raw baseline")
        ap.add_argument("--target", default="detection", choices=sorted(join.TARGETS))
        ap.add_argument("--force", action="store_true")
        a = ap.parse_args(rest)
        arts = [_resolve(s) for s in a.artifacts.split(",") if s]
        join.run(arts, target=a.target, force=a.force)
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
        ap.add_argument("--force", action="store_true")
        a = ap.parse_args(rest)
        r = experiment.chain(smoothing_method=a.smoothing,
                             cpd_methods=a.cpd.split(",") if a.cpd else None,
                             on=a.on, model=a.model, target=a.target, force=a.force)
        return _show(r["evaluate"].read_table("metrics"))

    if cmd == "sweep":
        from wearable import experiment
        which = rest[0] if rest else "smoothing"
        opts = _knobs(rest[1:])
        fn = {"smoothing": experiment.sweep_smoothing,
              "cpd": experiment.sweep_cpd,
              "signal": experiment.sweep_signal,
              "smoothers-no-model": experiment.compare_smoothers_no_model}.get(which)
        if fn is None:
            raise SystemExit("sweep takes: smoothing | cpd | signal | smoothers-no-model")
        print(f"\n{'=' * 70}\nCOMPARISON: {which}\n{'=' * 70}")
        return _show(fn(**opts))

    raise SystemExit(f"unknown command {cmd!r}\n{__doc__}")


if __name__ == "__main__":
    main()
