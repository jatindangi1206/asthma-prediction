"""Step: forecast. Runs one frozen forecaster over a cohort and records the surprise.

    input   the ingest files of one channel
    output  one long table `forecasted`:
            p_id, createdTime, <value>, minute_diff, predicted_<sig>, surprise_<sig>,
            tail_<sig>, gap_flag, chunk_id

    python -m wearable forecast timestral --signal hr
    python -m wearable detect --on surprise --forecasting timestral --signal hr
    python -m wearable join --artifacts forecast:timestral --target onset

Deliberately the mirror image of `smooth`, and deliberately NOT part of it. A smoother
estimates the level AT t from data up to t; a forecaster predicts t+h from data up to t.
Their residuals therefore mean different things — noise in one case, SURPRISE in the
other — so putting them in one pool would make `sweep smoothing` compare two different
quantities, and rule 5 says a comparison varies only the method.

Three output columns, not one:

    predicted_<sig>  what the frozen model expected
    surprise_<sig>   observed - predicted, in the units of the signal
    tail_<sig>       where the observed value fell in the predictive distribution,
                     in [0, 1]; NaN for a method that returns no quantiles

`tail` is the one the smoothing contract could never carry, and the better signal of the
two: "this reading fell at the 99.5th percentile of what the model expected" is
calibrated and comparable across patients and channels, while "6 ms off" means different
things at a 30 ms and a 90 ms baseline.

Depends on no other step. Runs single-process by default: these are 120M-209M parameter
models and `cpu_count-1` workers would each load their own copy.
"""
from __future__ import annotations

import hashlib

import pandas as pd
from tqdm import tqdm

from wearable import registry, store
from wearable.config import (CHANNELS, DEFAULT_CHANNEL, SMOOTHABLE, PROCESSED_DIR, processed_files,
                             processed_pid)

STEP = "forecast"


def columns(signal: str = DEFAULT_CHANNEL):
    """-> (value, predicted, surprise, tail, the output column order)."""
    val = CHANNELS[signal]
    pred, sur, tail = f"predicted_{signal}", f"surprise_{signal}", f"tail_{signal}"
    return val, pred, sur, tail, ["p_id", "createdTime", val, "minute_diff", pred,
                                  sur, tail, "gap_flag", "chunk_id"]


def _one(file_path, method: str, signal: str, overrides: dict):
    """Forecast one patient file. Returns a frame or a reason it could not."""
    from wearable.forecasting import LEVELS
    from wearable.forecasting._base import forecast_frame

    pid = processed_pid(file_path)
    val, pred, sur, tail, cols = columns(signal)
    try:
        spec = registry.get("forecasting", method)
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        if not {"createdTime", val, "minute_diff"}.issubset(df.columns):
            return pid, None, f"missing columns, found {list(df.columns)}"

        df["createdTime"] = pd.to_datetime(df["createdTime"])
        obs = df[df[val].notna()].copy()
        obs["patient_id"] = pid
        cfg = spec.config(**overrides)
        levels = getattr(spec.load(), "LEVELS", LEVELS)
        fn = spec.entry()

        out = forecast_frame(obs, lambda c, h: fn(c, h, config=cfg, levels=levels),
                             levels, "patient_id", "createdTime", val, cfg,
                             pred, tail, sur)
        res = df.merge(out[["createdTime", pred, tail, "chunk_id"]],
                       on="createdTime", how="left")
        res["gap_flag"] = res[val].isna().astype(int)
        res[sur] = res[val] - res[pred]
        res["p_id"] = pid

        if int(res[pred].notna().sum()) == 0:
            return pid, None, "no chunk long enough to forecast"
        mn = float(res[pred].min(skipna=True))
        if not (mn > 0):                      # H1
            return pid, None, f"non-positive prediction ({mn})"
        return pid, res[cols], None
    except Exception as e:
        return pid, None, f"{type(e).__name__}: {e}"


def run(method: str, signal: str = DEFAULT_CHANNEL, force: bool = False,
        limit: int | None = None, patients: list[str] | None = None, **overrides):
    """`limit` caps the patient count and is part of the key, as in `smooth`."""
    if signal not in SMOOTHABLE:
        raise ValueError(
            f"forecast needs a continuous, positive channel. {signal!r} is "
            f"{'not a channel at all' if signal not in CHANNELS else 'a state flag or a sparse duration'}; "
            f"choose one of {list(SMOOTHABLE)}. The others are ingested and become "
            f"day features in `join`, but have no level to estimate (H1)")
    spec = registry.get("forecasting", method)
    params = {**spec.params(**overrides), "signal": signal}
    name = method if signal == DEFAULT_CHANNEL else f"{method}@{signal}"
    if limit:
        params = {**params, "limit_patients": int(limit)}
    if patients is not None:
        ids = sorted(set(map(str, patients)))
        params = {**params,
                  "patients_sha": hashlib.sha256(",".join(ids).encode()
                                                 ).hexdigest()[:12],
                  "n_patients_requested": len(ids)}
    inputs = [{"step": "ingest", "method": f"processed_dir:{signal}",
               "key": str(PROCESSED_DIR)}]

    if not force:
        hit = store.find(STEP, name, params, inputs, spec.source_hash)
        if hit is not None:
            print(f"[forecast/{name}] reusing {hit.key[:12]} "
                  f"({hit.manifest['stats'].get('n_patients')} patients)")
            return hit

    files = processed_files(signal)
    if patients is not None:
        files = [f for f in files if processed_pid(f) in set(ids)]
    if limit:
        files = files[:int(limit)]
    if not files:
        raise FileNotFoundError(
            f"no {signal} files in {PROCESSED_DIR} — run `ingest` first")

    art = store.open_artifact(STEP, name, params, inputs, spec.source_hash,
                              label=spec.label)
    print(f"\n[forecast/{name}] {spec.label}  signal={signal}")
    print(f"  params  {params}")
    print(f"  input   {len(files)} patients from {PROCESSED_DIR}")
    print(f"  key     {art.key[:12]}")

    # Single process on purpose: one copy of a 209M-parameter model, loaded once.
    frames, failures = [], []
    for f in tqdm(files, desc=f"forecast {name}"):
        pid, frame, err = _one(f, method, signal, overrides)
        (failures.append((pid, err)) if frame is None else frames.append(frame))

    if not frames:
        art.fail("every patient failed")
        raise RuntimeError(f"forecast/{name}: every patient failed\n"
                           + "\n".join(f"  {p}: {e}" for p, e in failures[:5]))

    table = pd.concat(frames, ignore_index=True).sort_values(["p_id", "createdTime"])
    art.write_table("forecasted", table.reset_index(drop=True))
    if failures:
        art.write_table("failures", pd.DataFrame(failures, columns=["p_id", "reason"]))

    _, _, sur, tail, _ = columns(signal)
    art.complete(signal=signal, n_patients=len(frames), n_failed=len(failures),
                 n_rows=len(table),
                 mean_abs_surprise=float(table[sur].abs().mean(skipna=True)),
                 has_tail=bool(table[tail].notna().any()))
    print(f"  -> {len(frames)} patients, {len(table):,} rows, {len(failures)} failed "
          f"({art.dir})")
    return art
