"""Wearable time-series research pipeline: signal -> smoothing -> CPD -> model -> metrics.

Five independent steps. None imports another; `wearable.experiment` is the only thing
that connects them, so any step runs on its own and they compose in any order:

    sources ─▶ ingest ─▶ smooth ──┐
                   │              ├─▶ join ─▶ model ─▶ evaluate
                   └─▶ detect ────┘
                       (on raw, smoothed or residual)

Every pool is discovered from the filesystem, so adding a method is **one file and
nothing else** — no list to register in:

    wearable/sources.py      a data format     patients() + load(pid)
    wearable/smoothing/      a smoother        smooth_dataframe(...)
    wearable/cpd/            a detector        detect(time, values, priors)
    wearable/models/         a model           fit(...) + predict_proba(...)
    wearable/metrics/        a metric          compute(...) + KIND

Every result is an artifact under `data/runs/<step>/<method>/<key>/` with a
`manifest.json` recording the method, its settings, its input artifacts and the code
version that produced it — so any result traces back to the raw signal
(`python -m wearable lineage ...`). The key is a hash of exactly those things, so
re-running the same computation reuses the stored result instead of repeating it, and
editing a method invalidates only that method.

Comparisons hold everything but the method fixed: the train/test split is derived from
the patient universe and the seed alone, never from feature content, so every variant
in a sweep sees the same test participants.
"""
__all__ = ["config", "chunking", "registry", "store", "sources", "smoothing", "cpd",
           "models", "metrics", "steps", "experiment", "selftest"]
