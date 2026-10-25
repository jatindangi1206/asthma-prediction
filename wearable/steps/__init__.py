"""The five steps. Each is independent: it takes artifacts and arguments, and writes an
artifact. None imports another — `wearable.experiment` wires them.

    ingest     a registered source   -> data/processed/ (+ labels)
    smooth     one smoothing method  -> table `smoothed`  (adds residual_hrv)
    detect     CPD on raw | smoothed | residual -> table `detected`
    join       any artifacts + labels -> tables `train`, `test` (patient-level split)
    model      a join artifact       -> table `predictions`
    evaluate   predictions or a smooth artifact -> table `metrics`

Every step reuses a cached artifact when method, settings, input and code version all
match; pass `force=True` to recompute.
"""
__all__ = ["ingest", "smooth", "detect", "join", "model", "evaluate"]
