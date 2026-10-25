"""Step: smoothing. Runs one discovered smoothing method over a cohort.

    input   an ingest artifact (per-patient files in data/processed/)
    output  one long table `smoothed`:
            p_id, createdTime, hrvValue, minute_diff, smoothed_hrv, residual_hrv,
            gap_flag, chunk_id

Depends on no other step. `residual_hrv = hrvValue - smoothed_hrv` is computed here
because it is a property of smoothing, not an architecture: the detect step chooses
whether to run on raw, smoothed or residual.
"""
from __future__ import annotations

import concurrent.futures
import multiprocessing
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from wearable import registry, store
from wearable.config import PROCESSED_DIR

warnings.filterwarnings("ignore")

STEP = "smooth"
REQUIRED = {"createdTime", "hrvValue", "minute_diff"}
COLS = ["p_id", "createdTime", "hrvValue", "minute_diff", "smoothed_hrv",
        "residual_hrv", "gap_flag", "chunk_id"]


def _one(args):
    """Smooth one patient file. Pure: takes paths and names, returns a frame."""
    file_path, method, overrides = args
    pid = Path(file_path).stem.replace("_processed", "")
    try:
        spec = registry.get("smoothing", method)
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        if not REQUIRED.issubset(df.columns):
            return pid, None, f"missing columns, found {list(df.columns)}"

        df["createdTime"] = pd.to_datetime(df["createdTime"])
        # H8: segment on observed readings only — the NaN gap fillers would otherwise
        # hide the real voids from the chunker.
        obs = df[df["hrvValue"].notna()].copy()
        obs["patient_id"] = pid

        smoothed = spec.entry()(obs, patient_col="patient_id",
                                timestamp_col="createdTime", value_col="hrvValue",
                                config=spec.config(**overrides),
                                out_col="smoothed_hrv")

        res = df.merge(smoothed[["createdTime", "smoothed_hrv", "chunk_id"]],
                       on="createdTime", how="left")
        res["gap_flag"] = res["hrvValue"].isna().astype(int)
        res["residual_hrv"] = res["hrvValue"] - res["smoothed_hrv"]
        res["p_id"] = pid

        if int(res["smoothed_hrv"].notna().sum()) == 0:
            return pid, None, "no chunk long enough to smooth"
        mn = float(res["smoothed_hrv"].min(skipna=True))
        if not (mn > 0):                      # H1
            return pid, None, f"non-positive smoothed value ({mn})"
        return pid, res[COLS], None
    except Exception as e:
        return pid, None, f"{type(e).__name__}: {e}"


def run(method: str, source_artifact=None, force: bool = False,
        workers: int | None = None, limit: int | None = None, **overrides):
    """`limit` caps the number of patients (first N by id) — for quick sweeps. It is
    part of the cache key, so a 20-patient run is never reused as a full-cohort one."""
    spec = registry.get("smoothing", method)
    params = spec.params(**overrides)
    if limit:
        params = {**params, "limit_patients": int(limit)}
    inputs = [source_artifact.ref] if source_artifact is not None else [
        {"step": "ingest", "method": "processed_dir", "key": str(PROCESSED_DIR)}]

    if not force:
        hit = store.find(STEP, method, params, inputs, spec.source_hash)
        if hit is not None:
            print(f"[smooth/{method}] reusing {hit.key[:12]} "
                  f"({hit.manifest['stats'].get('n_patients')} patients)")
            return hit

    files = sorted(PROCESSED_DIR.glob("*_processed.csv"))
    if limit:
        files = files[:int(limit)]
    if not files:
        raise FileNotFoundError(f"no processed CSVs in {PROCESSED_DIR} — run ingest first")

    art = store.open_artifact(STEP, method, params, inputs, spec.source_hash,
                              label=spec.label)
    workers = workers or max(1, multiprocessing.cpu_count() - 1)
    print(f"\n[smooth/{method}] {spec.label}")
    print(f"  params  {params}")
    print(f"  input   {len(files)} patients from {PROCESSED_DIR}")
    print(f"  key     {art.key[:12]}   workers {workers}")

    frames, failures = [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, (f, method, overrides)) for f in files]
        for fut in tqdm(concurrent.futures.as_completed(futures),
                        total=len(files), desc=f"smooth {method}"):
            pid, frame, err = fut.result()
            (failures.append((pid, err)) if frame is None else frames.append(frame))

    if not frames:
        art.fail("every patient failed")
        raise RuntimeError(f"smooth/{method}: every patient failed")

    table = pd.concat(frames, ignore_index=True).sort_values(["p_id", "createdTime"])
    art.write_table("smoothed", table.reset_index(drop=True))
    if failures:
        art.write_table("failures", pd.DataFrame(failures, columns=["p_id", "reason"]))

    art.complete(n_patients=len(frames), n_failed=len(failures), n_rows=len(table))
    print(f"  -> {len(frames)} patients, {len(table):,} rows, {len(failures)} failed "
          f"({art.dir})")
    return art
