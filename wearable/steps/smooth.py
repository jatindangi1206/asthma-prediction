"""Step: smoothing. Runs one discovered smoothing method over a cohort.

    input   an ingest artifact (per-patient files in data/processed/)
    output  one long table `smoothed`:
            p_id, createdTime, <value>, minute_diff, smoothed_<sig>, residual_<sig>,
            gap_flag, chunk_id

Depends on no other step. `residual_<sig> = <value> - smoothed_<sig>` is computed here
because it is a property of smoothing, not an architecture: the detect step chooses
whether to run on raw, smoothed or residual.

ANY CHANNEL, SAME TEN METHODS. `--signal hr` smooths heart rate instead of HRV with no
change to any smoother: `smooth_dataframe` already takes `value_col` as an argument, so
the pool was always channel-agnostic — only this step had HRV written into it. Each
channel reads its own ingest files, so H8 segments it on its own observation times.

`--signal hrv` is the default and is keyed and named exactly as before, so every
artifact and report built before the step became multi-channel is still reused.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import multiprocessing
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from wearable import registry, store
from wearable.config import (CHANNELS, DEFAULT_CHANNEL, SMOOTHABLE, PROCESSED_DIR, processed_files,
                             processed_pid)

warnings.filterwarnings("ignore")

STEP = "smooth"


def columns(signal: str = DEFAULT_CHANNEL):
    """-> (value column, smoothed column, residual column, the output column order)."""
    val = CHANNELS[signal]
    out, res = f"smoothed_{signal}", f"residual_{signal}"
    return val, out, res, ["p_id", "createdTime", val, "minute_diff", out, res,
                           "gap_flag", "chunk_id"]


# Kept for the HRV default so existing callers and tests read the same names.
REQUIRED = {"createdTime", "hrvValue", "minute_diff"}
COLS = columns(DEFAULT_CHANNEL)[3]


def _one(args):
    """Smooth one patient file. Pure: takes paths and names, returns a frame."""
    file_path, method, overrides, signal = args
    pid = processed_pid(file_path)
    val, out_col, res_col, cols = columns(signal)
    try:
        spec = registry.get("smoothing", method)
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        if not {"createdTime", val, "minute_diff"}.issubset(df.columns):
            return pid, None, f"missing columns, found {list(df.columns)}"

        df["createdTime"] = pd.to_datetime(df["createdTime"])
        # H8: segment on observed readings only — the NaN gap fillers would otherwise
        # hide the real voids from the chunker.
        obs = df[df[val].notna()].copy()
        obs["patient_id"] = pid

        smoothed = spec.entry()(obs, patient_col="patient_id",
                                timestamp_col="createdTime", value_col=val,
                                config=spec.config(**overrides),
                                out_col=out_col)

        res = df.merge(smoothed[["createdTime", out_col, "chunk_id"]],
                       on="createdTime", how="left")
        res["gap_flag"] = res[val].isna().astype(int)
        res[res_col] = res[val] - res[out_col]
        res["p_id"] = pid

        if int(res[out_col].notna().sum()) == 0:
            return pid, None, "no chunk long enough to smooth"
        mn = float(res[out_col].min(skipna=True))
        if not (mn > 0):                      # H1
            return pid, None, f"non-positive smoothed value ({mn})"
        return pid, res[cols], None
    except Exception as e:
        return pid, None, f"{type(e).__name__}: {e}"


def run(method: str, source_artifact=None, force: bool = False,
        workers: int | None = None, limit: int | None = None,
        patients: list[str] | None = None, signal: str = DEFAULT_CHANNEL,
        **overrides):
    """`limit` caps the number of patients (first N by id) — for quick sweeps. It is
    part of the cache key, so a 20-patient run is never reused as a full-cohort one.
    `patients` restricts the run to those ids (a cohort); its fingerprint is keyed too.
    `signal` picks the channel (`config.CHANNELS`)."""
    if signal not in SMOOTHABLE:
        raise ValueError(
            f"smooth needs a continuous, positive channel. {signal!r} is "
            f"{'not a channel at all' if signal not in CHANNELS else 'a state flag or a sparse duration'}; "
            f"choose one of {list(SMOOTHABLE)}. The others are ingested and become "
            f"day features in `join`, but have no level to estimate (H1)")
    spec = registry.get("smoothing", method)
    params = spec.params(**overrides)
    # Keyed and named only when it is not the default, so every HRV artifact built
    # before this step became multi-channel keeps its key and is still reused.
    name = method if signal == DEFAULT_CHANNEL else f"{method}@{signal}"
    if signal != DEFAULT_CHANNEL:
        params = {**params, "signal": signal}
    if limit:
        params = {**params, "limit_patients": int(limit)}
    if patients is not None:
        ids = sorted(set(map(str, patients)))
        params = {**params, "patients_sha": hashlib.sha256(",".join(ids).encode()).hexdigest()[:12],
                  "n_patients_requested": len(ids)}
    inputs = [source_artifact.ref] if source_artifact is not None else [
        {"step": "ingest", "method": "processed_dir", "key": str(PROCESSED_DIR)}]

    if not force:
        hit = store.find(STEP, name, params, inputs, spec.source_hash)
        if hit is not None:
            print(f"[smooth/{name}] reusing {hit.key[:12]} "
                  f"({hit.manifest['stats'].get('n_patients')} patients)")
            return hit

    files = processed_files(signal)
    if patients is not None:
        files = [f for f in files if processed_pid(f) in set(ids)]
    if limit:
        files = files[:int(limit)]
    if not files:
        raise FileNotFoundError(
            f"no {signal} files in {PROCESSED_DIR} — run `ingest` first"
            + ("" if signal == DEFAULT_CHANNEL else
               f" (it writes <pid>__{signal}_processed.csv)"))

    art = store.open_artifact(STEP, name, params, inputs, spec.source_hash,
                              label=spec.label)
    workers = workers or max(1, multiprocessing.cpu_count() - 1)
    print(f"\n[smooth/{name}] {spec.label}  signal={signal}")
    print(f"  params  {params}")
    print(f"  input   {len(files)} patients from {PROCESSED_DIR}")
    print(f"  key     {art.key[:12]}   workers {workers}")

    frames, failures = [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, (f, method, overrides, signal)) for f in files]
        for fut in tqdm(concurrent.futures.as_completed(futures),
                        total=len(files), desc=f"smooth {name}"):
            pid, frame, err = fut.result()
            (failures.append((pid, err)) if frame is None else frames.append(frame))

    if not frames:
        art.fail("every patient failed")
        raise RuntimeError(f"smooth/{name}: every patient failed")

    table = pd.concat(frames, ignore_index=True).sort_values(["p_id", "createdTime"])
    art.write_table("smoothed", table.reset_index(drop=True))
    if failures:
        art.write_table("failures", pd.DataFrame(failures, columns=["p_id", "reason"]))

    art.complete(signal=signal, n_patients=len(frames), n_failed=len(failures),
                 n_rows=len(table))
    print(f"  -> {len(frames)} patients, {len(table):,} rows, {len(failures)} failed "
          f"({art.dir})")
    return art
