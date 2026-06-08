#!/usr/bin/env python3
"""
src/patch_add_trend_level.py  —  Backfill true_trend_level and gap_flag into
existing smoothed CSVs that were produced before the 02_run_filters.py fix.

Method:
  true_trend_level ≈ 24-hour centred moving average of smoothed_hrv.
  A symmetric MA with window = 144 steps (= 24 h at 10-min sampling)
  cancels the 24-hour circadian component (c1) and the 12-hour semi-diurnal
  component (c2) exactly, because 72 divides 144.  The result is a smooth
  baseline identical in character to the particle smoother's level state,
  without needing to re-run the expensive FFBS.

  This is an APPROXIMATION.  It differs slightly from the exact particle
  smoother level near rapid baseline changes and at segment edges.  Run
  02_run_filters.py for the exact values when you have time.

Usage (from asthma-prediction/):
    python src/patch_add_trend_level.py            # all patients
    python src/patch_add_trend_level.py 0010       # single patient
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SMOOTHED_DIR = Path("./data/smoothed")
LAGS_DAY     = 144    # 24 h at 10-min sampling


def patch_file(fpath: Path) -> dict:
    stem = fpath.stem.replace("_smoothed", "")
    df   = pd.read_csv(fpath, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    if "smoothed_hrv" not in df.columns:
        return {"file": fpath.name, "status": "skipped", "reason": "no smoothed_hrv column"}

    already_has = ("true_trend_level" in df.columns and "gap_flag" in df.columns)

    # ── gap_flag ──────────────────────────────────────────────────────────
    # 1 where the original HRV observation was missing (gap-fill placeholder row)
    if "gap_flag" not in df.columns:
        if "hrvValue" in df.columns:
            df["gap_flag"] = df["hrvValue"].isna().astype(int)
        else:
            df["gap_flag"] = 0

    # ── true_trend_level (approximation via 24-hour centred MA) ──────────
    if "true_trend_level" not in df.columns:
        df["true_trend_level"] = (
            df["smoothed_hrv"]
            .rolling(window=LAGS_DAY, center=True, min_periods=LAGS_DAY // 4)
            .mean()
        )
        # Near the edges (first/last ~72 rows), the MA window is incomplete.
        # Fill those with a shorter-window MA to avoid edge NaN.
        edge_mask = df["true_trend_level"].isna()
        if edge_mask.any():
            short_ma = df["smoothed_hrv"].rolling(
                window=LAGS_DAY // 4, center=True, min_periods=1
            ).mean()
            df.loc[edge_mask, "true_trend_level"] = short_ma[edge_mask]

    # ── reorder columns ───────────────────────────────────────────────────
    core = ["createdTime", "hrvValue", "minute_diff",
            "smoothed_hrv", "true_trend_level", "gap_flag"]
    extra = [c for c in df.columns if c not in core]
    df = df[core + extra]

    df.to_csv(fpath, index=False)

    n_gaps = int(df["gap_flag"].sum())
    ttl_std = float(df["true_trend_level"].std())
    smh_std = float(df["smoothed_hrv"].std())
    print(f"  ✓ {stem:>10}  rows={len(df)}  gap_rows={n_gaps}  "
          f"smoothed_std={smh_std:.1f}ms  trend_std={ttl_std:.1f}ms  "
          f"{'(already had both cols — refreshed)' if already_has else '(added)'}")

    return {"file": fpath.name, "status": "success", "n_rows": len(df),
            "n_gap_rows": n_gaps, "trend_std_ms": ttl_std}


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else None

    if target:
        files = list(SMOOTHED_DIR.glob(f"*{target}*_smoothed.csv"))
        if not files:
            sys.exit(f"[ERROR] No smoothed CSV matching '{target}' in {SMOOTHED_DIR.resolve()}")
    else:
        files = sorted(SMOOTHED_DIR.glob("*_smoothed.csv"))
        if not files:
            sys.exit(f"[ERROR] No *_smoothed.csv in {SMOOTHED_DIR.resolve()}")

    print(f"\n{'='*68}")
    print(f"  Backfill: true_trend_level + gap_flag  |  {len(files)} file(s)")
    print(f"  Method: 24-h centred moving-average of smoothed_hrv")
    print(f"  Note: approximate — re-run 02_run_filters.py for exact values")
    print(f"{'='*68}")

    results = [patch_file(f) for f in files]

    success = sum(1 for r in results if r["status"] == "success")
    print(f"\n  Done: {success}/{len(files)} files updated")
    print(f"  All smoothed CSVs now contain: smoothed_hrv, true_trend_level, gap_flag")


if __name__ == "__main__":
    main()
