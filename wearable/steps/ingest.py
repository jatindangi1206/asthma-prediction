"""Stage 0 — ingest: any registered source -> the one contract everything downstream reads.

  Output:  data/processed/<pid>_processed.csv   (createdTime, hrvValue, minute_diff)
           data/labels_pass1.csv                (day-level ground truth, labelled sources only)

This replaces the two hand-written Stage 0 scripts. The source-specific reading lives in
`wearable.sources`; the gap rule and the output contract live here, once.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from wearable import sources
from wearable.config import (BURN_IN_N, FILL_INTERVAL_MIN, GAP_THRESHOLD_MIN,
                           LABELS_CSV, LOGS_DIR, PROCESSED_DIR)


def fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, coerce, and write NaN-HRV filler rows across every long gap.

    A filler every FILL_INTERVAL_MIN keeps the shape of a void visible on the output grid
    without inventing readings. Smoothers and the CPD stage segment on *observed* rows, so
    the fillers never hide a void from the chunker.
    """
    df = df.copy()
    df["createdTime"] = pd.to_datetime(df["createdTime"], errors="coerce")
    df["hrvValue"] = pd.to_numeric(df["hrvValue"], errors="coerce")
    df = df.dropna(subset=["createdTime"]).sort_values("createdTime").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["createdTime", "hrvValue", "minute_diff"])

    # ponytail: strictly `>`, while the H8 chunker splits on `>=`. A gap of exactly
    # 180.0 min therefore opens a new chunk but gets no filler row. Preserved from the
    # original Stage 0 because changing it would move every downstream output; the
    # cadence makes an exact 180.000-min gap vanishingly rare in practice.
    gaps = (df["createdTime"].diff().dt.total_seconds() / 60.0) > GAP_THRESHOLD_MIN

    fillers = []
    for idx in df[gaps].index:
        times = pd.date_range(
            start=df.at[idx - 1, "createdTime"] + pd.Timedelta(minutes=FILL_INTERVAL_MIN),
            end=df.at[idx, "createdTime"],
            freq=f"{FILL_INTERVAL_MIN}min",
            inclusive="left",
        )
        fillers.append(pd.DataFrame({"createdTime": times, "hrvValue": np.nan}))

    if fillers:
        df = (pd.concat([df, *fillers], ignore_index=True)
                .sort_values("createdTime").reset_index(drop=True))

    df["minute_diff"] = df["createdTime"].diff().dt.total_seconds() / 60.0
    return df[["createdTime", "hrvValue", "minute_diff"]]


def _ingest_one(args: tuple[str, str]) -> tuple[str, str, str, pd.DataFrame | None]:
    source_key, pid = args
    src = sources.get(source_key)
    try:
        data = src.load(pid)
    except Exception as exc:                       # one bad file must not stop the cohort
        return pid, "failed", f"unreadable: {exc}", None

    if data.hrv.empty:
        return pid, "failed", data.note or "no HRV readings", data.labels

    out = fill_gaps(data.hrv)
    if out.empty:
        return pid, "failed", "no valid rows after gap filling", data.labels

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROCESSED_DIR / f"{pid}_processed.csv", index=False)

    real = int(out["hrvValue"].notna().sum())
    armed = "armed" if real >= BURN_IN_N else f"NEVER ARMS (<{BURN_IN_N} valid samples)"
    return pid, "success", f"rows={len(out)} real={real} nan={len(out) - real} {armed}", data.labels


def run(source_key: str = sources.DEFAULT_SOURCE, workers: int | None = None) -> None:
    src = sources.get(source_key)
    pids = src.patients()
    if not pids:
        print(f"ERROR: source {source_key!r} offers no patients")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)

    print(f"\n--- STAGE 0 INGEST — source = {source_key} ---")
    print(f"Source:   {src.label}")
    print(f"Output:   {PROCESSED_DIR.resolve()}")
    print(f"Patients: {len(pids)}   Workers: {workers}   Gap={GAP_THRESHOLD_MIN:g} min\n")

    results, label_frames = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for pid, status, detail, labels in tqdm(
                pool.map(_ingest_one, [(source_key, p) for p in pids]),
                total=len(pids), desc=f"ingest {source_key}"):
            results.append((pid, status, detail))
            if labels is not None:
                label_frames.append(labels)

    if src.has_labels and label_frames:
        pd.concat(label_frames, ignore_index=True).to_csv(LABELS_CSV, index=False)

    log = pd.DataFrame(results, columns=["patient", "status", "detail"]).sort_values("patient")
    log.to_csv(log_path := LOGS_DIR / f"ingest_{source_key}_log.csv", index=False)

    fails = log[log["status"] == "failed"]
    never = int(log["detail"].str.contains("NEVER ARMS").sum())
    print("\n" + "=" * 64 + f"\nINGEST COMPLETE — {source_key}\n" + "=" * 64)
    print(f"Written:   {len(log) - len(fails)} / {len(log)} patients")
    print(f"Failed:    {len(fails)}")
    print(f"Won't arm: {never} (<{BURN_IN_N} valid samples — annotate emits nothing, by design)")
    if src.has_labels and label_frames:
        print(f"Labels:    {LABELS_CSV.resolve()}")
    print(f"Log:       {log_path.resolve()}")
    if not fails.empty:
        reasons = fails["detail"].value_counts()
        print("\nFailures by reason:")
        for reason, n in reasons.items():
            print(f"  {n:>4}  {reason}")
