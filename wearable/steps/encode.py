"""Step: encode. Turns each patient-day's window into a representation vector.

    input   the ingest files of one channel, or a smooth / forecast artifact
    output  one table `embedded`:  p_id, date, e0000 ... e<d-1>

    python -m wearable encode moment --signal hr --days 7 --profile 73
    python -m wearable join --artifacts encode:moment --target detection

WHY THIS IS A STEP AND NOT PART OF `model`
------------------------------------------
A frozen forward pass does not depend on the fold, the seed, the head or the task. Run
inside a model it is recomputed for every (model x fold x seed) cell and its output can
reach neither `cpd/` nor the other task. Run here it is computed ONCE, content-keyed by
the artifact store, and reused by every fold, every seed, every model and both tasks —
which is the only reason a 0.3B encoder is affordable on this cohort at all.

It also keeps rule 1 intact: this step depends on no other. It is handed a channel or an
artifact, builds the window with the shared `wearable/windowing.py` rule, and writes a
table that `join` merges like any other.

GRAIN: this artifact holds ONE ROW PER PATIENT-DAY, not one per reading, so the manifest
records `grain="day"` and `join` passes the columns straight through instead of
aggregating them to mean/std/min/max. Without that flag a 1,024-wide embedding would
become 4,096 columns, three quarters of them a standard deviation over a single row.

NO REDUCTION HERE, DELIBERATELY. The full vector is written and PCA or any other
reduction happens inside the model, where the train/test split exists. Reducing here
would fit on test patients, which is a leak — this step runs before the split is drawn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wearable import encoders, registry, store
from wearable.config import (CHANNELS, DAYS, DEFAULT_CHANNEL, PROFILE_BINS,
                             PROCESSED_DIR, SMOOTHABLE, processed_files,
                             processed_pid)
from wearable.windowing import day_profile, lag_window, ordered_series

STEP = "encode"


def _windows(signal: str, days: int, profile: int, input_artifact=None):
    """-> (the (p_id, date) index, the (n, days*profile) window matrix).

    One row per patient-day that has any reading, with the preceding `days-1` days of
    history beside it — the same window rule `join` uses, imported, not re-implemented.
    """
    if input_artifact is None:
        frames = []
        for f in processed_files(signal):
            d = pd.read_csv(f, encoding="utf-8-sig")
            d["p_id"] = processed_pid(f)
            frames.append(d)
        if not frames:
            raise FileNotFoundError(
                f"no {signal} files in {PROCESSED_DIR} — run `ingest` first")
        table = pd.concat(frames, ignore_index=True)
        prefix = f"raw_{signal}"
    else:
        table = input_artifact.read_table(input_artifact.manifest["tables"][0])
        prefix = f"{input_artifact.step[:2]}_{input_artifact.method}".replace("+", "_")

    table["p_id"] = table["p_id"].astype(str)
    table["createdTime"] = pd.to_datetime(table["createdTime"])
    table["date"] = table["createdTime"].dt.normalize()

    prof = day_profile(table, prefix, int(profile))
    if prof is None:
        raise ValueError(f"nothing to profile in the input for signal={signal!r}")
    windowed = lag_window(prof.reset_index(), int(days))
    index = windowed[["p_id", "date"]].reset_index(drop=True)
    return index, ordered_series(windowed)


def run(method: str, signal: str = DEFAULT_CHANNEL, days: int = DAYS,
        profile: int = PROFILE_BINS, input_artifact=None, force: bool = False,
        **overrides):
    """`days * profile` is the context the encoder receives — see `wearable/windowing`.

    A method with a fixed input length (`CONTEXT`) resamples onto it; pass
    `--mode decimate` or `--mode interpolate` to choose how, and
    `windowing.bins_for(days, CONTEXT)` to avoid needing either.
    """
    if signal not in SMOOTHABLE:
        raise ValueError(
            f"encode needs a continuous, positive channel. {signal!r} is "
            f"{'not a channel at all' if signal not in CHANNELS else 'a state flag or a sparse duration'}; "
            f"choose one of {list(SMOOTHABLE)}. The others are ingested and become "
            f"day features in `join`, but have no level to estimate (H1)")
    spec = registry.get("encoders", method)
    params = {**spec.params(**overrides), "signal": signal,
              "days": int(days), "profile": int(profile)}
    inputs = ([input_artifact.ref] if input_artifact is not None else
              [{"step": "ingest", "method": f"processed_dir:{signal}",
                "key": str(PROCESSED_DIR)}])

    if not force:
        hit = store.find(STEP, method, params, inputs, spec.source_hash)
        if hit is not None:
            print(f"[encode/{method}] reusing {hit.key[:12]} "
                  f"({hit.manifest['stats'].get('n_rows')} patient-days)")
            return hit

    ctx = int(days) * int(profile)
    ceiling = getattr(spec.load(), "CONTEXT", None)
    print(f"\n[encode/{method}] {spec.label}  signal={signal}")
    print(f"  window  {days} days x {profile} slots = {ctx} points"
          + (f"; model takes {ceiling} (mode={params.get('mode')})" if ceiling else ""))

    index, series = _windows(signal, days, profile, input_artifact)
    print(f"  input   {len(index):,} patient-days, "
          f"{100 * np.isfinite(series).mean():.1f}% of slots observed")

    art = store.open_artifact(STEP, method, params, inputs, spec.source_hash,
                              label=spec.label)
    emb = np.asarray(spec.entry()(series, **spec.params(**overrides)), dtype=np.float32)
    if emb.ndim != 2 or len(emb) != len(index):
        art.fail(f"encoder returned {emb.shape} for {len(index)} windows")
        raise RuntimeError(f"encode/{method}: expected ({len(index)}, d), got {emb.shape}")

    table = pd.concat(
        [index, pd.DataFrame(emb, columns=[f"e{i:04d}" for i in range(emb.shape[1])])],
        axis=1)
    art.write_table("embedded", table)
    # `grain` is what tells join these rows are already per patient-day.
    art.manifest["grain"] = "day"
    art.complete(grain="day", signal=signal, days=int(days), profile=int(profile),
                 context=ctx, dim=int(emb.shape[1]), n_rows=len(table),
                 n_patients=int(index["p_id"].nunique()),
                 observed_share=float(np.isfinite(series).mean()))
    print(f"  -> {len(table):,} patient-days x {emb.shape[1]} dims "
          f"(key {art.key[:12]})")
    return art
