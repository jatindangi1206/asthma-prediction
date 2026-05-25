"""Stage 4 — run all four detectors on smoothed HRV and assemble the final CSV.

Reads ``<id>_smoothed.csv``, runs bocpd / kcp / hmm / kalman on the smoothed HRV
only, and writes ``pipeline/outputs/<id>_final.csv``. Each detector contributes
its OWN ``*_indicator`` (0/1) and ``*_magnitude`` ([0,1]) columns — never merged
or averaged. ALL context columns from the input frame are carried through
unchanged (the prime directive): after building the detector frame, every column
that isn't a remapped/internal working column is appended.

    python -m pipeline.cpd.cpd_pipeline <patient_id>
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline import config
from pipeline.cpd import bocpd, kcp, hmm, kalman

DETECTORS = [
    ("bocpd", bocpd.detect, "tab:blue"),
    ("kcp", kcp.detect, "tab:orange"),
    ("hmm", hmm.detect, "tab:green"),
    ("kalman", kalman.detect, "tab:red"),
]

# Context order for the final CSV tail (anything else present is still carried).
CONTRACT_TAIL = [
    "HR", "steps", "temperature", "spo2", "is_asleep",
    "hr_mask", "steps_mask", "temp_mask", "spo2_mask",
    "watch_off_flag", "clip_flag", "timestamp",
]


def _indicator_from_types(change_types: np.ndarray) -> np.ndarray:
    """0/1 indicator from a detector's string-typed output ('normal' => 0)."""
    return (np.asarray(change_types, dtype=object) != "normal").astype(int)


def run_cpd(patient_id: str, make_plot: bool = True) -> Path:
    config.ensure_dirs()
    src = config.processed_path(patient_id, "smoothed")
    if not src.exists():
        raise FileNotFoundError(f"Run Stage 3 first; missing {src}")
    print(f"=== Stage 4: change-point detection {patient_id} ===")

    df = pd.read_csv(src)
    time_col, real_col, smoothed_col = (
        config.MINUTE_COL, config.HRV_RAW_COL, config.SMOOTHED_COL)

    time_full = pd.to_numeric(df[time_col], errors="coerce").to_numpy(float)
    smoothed_full = pd.to_numeric(df[smoothed_col], errors="coerce").to_numpy(float)
    valid = np.isfinite(time_full) & np.isfinite(smoothed_full)
    if not valid.any():
        raise ValueError(f"No finite smoothed HRV in {src.name}")
    time_valid, smoothed_valid = time_full[valid], smoothed_full[valid]

    # --- detector frame: timestep, real_value, smoothed_value + per-detector cols
    out = pd.DataFrame({
        "timestep": df[time_col].to_numpy(),
        "real_value": df[real_col].to_numpy(),
        "smoothed_value": df[smoothed_col].to_numpy(),
    })

    plot_points = []
    for name, detect_fn, color in DETECTORS:
        indicator = np.zeros(len(df), dtype=int)
        magnitude = np.zeros(len(df), dtype=float)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                types_v, degrees_v = detect_fn(time_valid, smoothed_valid)
            indicator[valid] = _indicator_from_types(types_v)
            magnitude[valid] = np.clip(np.asarray(degrees_v, dtype=float), 0.0, 1.0)
        except Exception as exc:  # one detector failing must not drop the others
            print(f"  [{name}] failed ({exc}); columns set to 0")
        out[f"{name}_indicator"] = indicator
        out[f"{name}_magnitude"] = magnitude
        print(f"  [{name}] {int(indicator.sum())} change points")
        plot_points.append((name, time_full[indicator.astype(bool)],
                            smoothed_full[indicator.astype(bool)], color))

    # --- carry through EVERY context column (the prime directive) -------------
    excluded = {time_col, real_col, smoothed_col,
                config.HRV_CLEAN_COL, config.MINUTE_DIFF_COL, config.MEAN_GAP_COL}
    carried = [c for c in df.columns if c not in excluded]
    ordered = ([c for c in CONTRACT_TAIL if c in carried]
               + [c for c in carried if c not in CONTRACT_TAIL])
    for c in ordered:
        out[c] = df[c].to_numpy()

    dest = config.final_path(patient_id)
    out.to_csv(dest, index=False)
    print(f"[{patient_id}] final -> {dest}  ({len(out)} rows, {len(out.columns)} cols)")

    if make_plot:
        _plot(patient_id, time_full, df[real_col].to_numpy(), smoothed_full, plot_points)
    return dest


def _plot(patient_id, time_full, real, smoothed, plot_points):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.figure(figsize=(14, 6))
    plt.plot(time_full, real, color="#9aa0a6", alpha=0.5, label="real")
    plt.plot(time_full, smoothed, color="black", linewidth=1.2, label="smoothed")
    for name, xs, ys, color in plot_points:
        if len(xs):
            plt.scatter(xs, ys, s=30, color=color, alpha=0.75, label=name.upper())
    plt.xlabel("timestep (min)"); plt.ylabel("HRV")
    plt.title(f"CPD overlay — {patient_id}")
    plt.legend(loc="best"); plt.tight_layout()
    png = config.OUTPUTS_DIR / f"{config._safe(patient_id)}_cpd.png"
    plt.savefig(png); plt.close()
    print(f"[{patient_id}] plot -> {png}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m pipeline.cpd.cpd_pipeline <patient_id>")
        raise SystemExit(2)
    run_cpd(sys.argv[1])
