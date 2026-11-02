"""Stage 0 — ingest: any registered source -> the one contract everything downstream reads.

  Output:  data/processed/<pid>_processed.csv             HRV   (createdTime, hrvValue, minute_diff)
           data/processed/<pid>__<channel>_processed.csv  every other channel
           data/processed/labels_pass1.csv               day-level ground truth

This replaces the two hand-written Stage 0 scripts. The source-specific reading lives in
`wearable.sources`; the gap rule and the output contract live here, once.

ONE FILE PER PATIENT-CHANNEL, not one file holding every channel. The channels differ
enormously in cadence — heart rate is on ~100% of raw rows and HRV on ~6% — and H8
segments a series on ITS OWN observation times. A shared grid would make a 180-minute
HRV hole invisible wherever heart rate happened to be recorded, so each channel gets
its own grid, its own filler rows and its own `minute_diff`. Downstream nothing changes:
`smooth --signal hr` reads the hr files with the same ten methods.

HRV keeps the original filename and byte-identical content, so no cached artifact,
report or key built before the pipeline became multi-channel is invalidated.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from wearable import sources
from wearable.config import (BURN_IN_N, CHANNELS, DEFAULT_CHANNEL, FILL_INTERVAL_MIN,
                             GAP_THRESHOLD_MIN, LABELS_CSV, LOGS_DIR, PROCESSED_DIR,
                             processed_path)


def fill_gaps(df: pd.DataFrame, value_col: str = "hrvValue") -> pd.DataFrame:
    """Sort, coerce, and write NaN-value filler rows across every long gap.

    A filler every FILL_INTERVAL_MIN keeps the shape of a void visible on the output grid
    without inventing readings. Smoothers and the CPD stage segment on *observed* rows, so
    the fillers never hide a void from the chunker.

    `value_col` makes this one implementation serve every channel. Rows where that
    channel has no reading are dropped first, so each channel's grid is built from its
    own observations — passing a multi-channel frame gives exactly the same result as
    passing a pre-filtered single-channel one.
    """
    df = df.copy()
    df["createdTime"] = pd.to_datetime(df["createdTime"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = (df.dropna(subset=["createdTime", value_col])
            .sort_values("createdTime").reset_index(drop=True))
    if df.empty:
        return pd.DataFrame(columns=["createdTime", value_col, "minute_diff"])

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
        fillers.append(pd.DataFrame({"createdTime": times, value_col: np.nan}))

    if fillers:
        df = (pd.concat([df, *fillers], ignore_index=True)
                .sort_values("createdTime").reset_index(drop=True))

    df["minute_diff"] = df["createdTime"].diff().dt.total_seconds() / 60.0
    return df[["createdTime", value_col, "minute_diff"]]


def _ingest_one(args: tuple[str, str, tuple]) -> tuple:
    """-> (pid, status, detail, labels, {channel: observed readings written})

    `status` stays the HRV verdict, so the cohort counts this step has always printed
    keep their meaning. A patient with heart rate but no HRV is still "failed" here and
    now gets an `hr` file anyway, which is the whole point of carrying the channels.
    """
    source_key, pid, want = args
    src = sources.get(source_key)
    try:
        data = src.load(pid)
    except Exception as exc:                       # one bad file must not stop the cohort
        return pid, "failed", f"unreadable: {exc}", None, {}

    present = data.channels()
    have = [c for c in present if c in want]
    if not have:
        # Distinguish "this patient has nothing" from "this patient has nothing we
        # asked for" — with --channels hrv the 30 heart-rate-only patients hit the
        # second, and calling that "no readings" is simply untrue.
        note = (data.note or "no readings in any channel") if not present else \
            f"no {'/'.join(want)} readings (has {', '.join(present)})"
        return pid, "failed", note, data.labels, {}

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}
    for ch in have:
        out = fill_gaps(data.readings, CHANNELS[ch])
        if out.empty:
            continue
        out.to_csv(processed_path(pid, ch), index=False)
        written[ch] = int(out[CHANNELS[ch]].notna().sum())

    extra = " ".join(f"{c}={n}" for c, n in written.items() if c != DEFAULT_CHANNEL)
    if DEFAULT_CHANNEL not in written:
        note = data.note or "no HRV readings"
        return pid, "failed", f"{note} (wrote {extra})" if extra else note, data.labels, written

    real = written[DEFAULT_CHANNEL]
    armed = "armed" if real >= BURN_IN_N else f"NEVER ARMS (<{BURN_IN_N} valid samples)"
    detail = f"real={real} {armed}" + (f" | {extra}" if extra else "")
    return pid, "success", detail, data.labels, written


def run(source_key: str = sources.DEFAULT_SOURCE, workers: int | None = None,
        channels=None) -> None:
    """`channels` restricts which signals are written (default: every one present)."""
    want = tuple(channels) if channels else tuple(CHANNELS)
    unknown = set(want) - set(CHANNELS)
    if unknown:
        raise ValueError(f"unknown channels {sorted(unknown)}; "
                         f"available: {sorted(CHANNELS)}")
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
    print(f"Patients: {len(pids)}   Workers: {workers}   Gap={GAP_THRESHOLD_MIN:g} min")
    print(f"Channels: {', '.join(want)}\n")

    results, label_frames = [], []
    per_channel: dict[str, int] = {c: 0 for c in want}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for pid, status, detail, labels, written in tqdm(
                pool.map(_ingest_one, [(source_key, p, want) for p in pids]),
                total=len(pids), desc=f"ingest {source_key}"):
            results.append((pid, status, detail, ",".join(written)))
            for c in written:
                per_channel[c] = per_channel.get(c, 0) + 1
            if labels is not None:
                label_frames.append(labels)

    if src.has_labels and label_frames:
        pd.concat(label_frames, ignore_index=True).to_csv(LABELS_CSV, index=False)

    log = pd.DataFrame(results, columns=["patient", "status", "detail", "channels"]
                       ).sort_values("patient")
    log.to_csv(log_path := LOGS_DIR / f"ingest_{source_key}_log.csv", index=False)

    fails = log[log["status"] == "failed"]
    never = int(log["detail"].str.contains("NEVER ARMS").sum())
    print("\n" + "=" * 64 + f"\nINGEST COMPLETE — {source_key}\n" + "=" * 64)
    print(f"Written:   {len(log) - len(fails)} / {len(log)} patients (HRV)")
    print(f"Failed:    {len(fails)} (no HRV; any other channel they have is still written)")
    print("Channels:  " + "  ".join(f"{c} {n}" for c, n in per_channel.items() if n))
    print(f"Won't arm: {never} (<{BURN_IN_N} valid samples — annotate emits nothing, by design)")
    if src.has_labels and label_frames:
        print(f"Labels:    {LABELS_CSV.resolve()}")
    print(f"Log:       {log_path.resolve()}")
    if not fails.empty:
        reasons = fails["detail"].value_counts()
        print("\nFailures by reason:")
        for reason, n in reasons.items():
            print(f"  {n:>4}  {reason}")
