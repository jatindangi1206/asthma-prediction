"""Stage 2c (alternative smoother): Kernel Recursive Least-Squares Tracker.

A NEW, independent smoothing method for benchmarking against the existing
`particles` SMC+FFBS smoother (`02_run_filters.py`) and the regime-switching
particle filter (`02b_run_rspf.py`). This script does NOT modify or read either
of those pipelines' outputs.

  Input :  data/processed/<pid>_processed.csv   (from 00_preprocess_raw.py)
  Output:  data/smoothed_krlst/<pid>_krlst.csv

Output columns:

  createdTime, hrvValue, minute_diff, smoothed_hrv, gap_flag, chunk_id

  * smoothed_hrv = KRLST causal estimate (single output, strictly > 0)
  * chunk_id     = H8 segment id (resets at >= 180-min gaps)

KRLST has no separate "trend level" / circadian-removed component (that belonged
to the particle-filter model); this method emits one smoothed signal.

To benchmark CPD on this method: point 03_annotate.py's INPUT_DIR at
data/smoothed_krlst (it falls back to smoothed_hrv when true_trend_level absent).

Run from the asthma-prediction/ directory:
    python src/02c_run_krlst.py
"""

import warnings
from pathlib import Path
import concurrent.futures
import multiprocessing

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Import the core smoother. Works whether run as `python src/02c_run_krlst.py`
# or `python -m src.02c_run_krlst`.
try:
    from krlst_smoother import smooth_dataframe, KRLSTConfig
except ImportError:  # pragma: no cover
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from krlst_smoother import smooth_dataframe, KRLSTConfig

INPUT_DIR = Path("./data/processed")
OUTPUT_DIR = Path("./data/smoothed_krlst")
RESULTS_DIR = Path("./data/results")

# ---- Method configuration ------------------------------------------------
GLOBAL_BUDGET = 100          # dictionary budget M (cost ~ O(M^2) per sample)
GLOBAL_LAMBDA = 0.995        # forgetting factor (<1 tracks the drifting baseline)


def process_patient(args):
    """Smooth one processed patient CSV with KRLST and write the output."""
    file_path, output_dir, budget, lam = args
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()

        if not {"createdTime", "hrvValue", "minute_diff"}.issubset(df.columns):
            return {"file": file_path.name, "status": "failed",
                    "reason": f"Missing columns. Found: {list(df.columns)}"}

        stem = file_path.stem.replace("_processed", "")
        df["createdTime"] = pd.to_datetime(df["createdTime"])

        # CRITICAL (H8): 00_preprocess_raw.py inserts NaN-hrv filler rows every
        # 10 min across gaps > 180 min. Those fillers make consecutive ROW times
        # look ~10 min apart, which would HIDE the real voids from the chunker.
        # The H8 rule is defined on consecutive *readings*, so we segment on
        # observed rows only: drop the fillers, let smooth_dataframe compute
        # time_diff between real readings, then merge results back onto the grid.
        obs = df[df["hrvValue"].notna()].copy()
        obs["patient_id"] = stem

        cfg = KRLSTConfig(budget=budget, forgetting_lambda=lam)
        smoothed = smooth_dataframe(
            obs,
            patient_col="patient_id",
            timestamp_col="createdTime",
            value_col="hrvValue",
            config=cfg,
            out_col="smoothed_hrv",
        )

        # Merge results back onto the full grid (keeps filler rows, NaN-smoothed,
        # for row-alignment parity with data/smoothed/<pid>_smoothed.csv).
        res = df.copy()
        merge_cols = ["createdTime", "smoothed_hrv", "chunk_id"]
        res = res.merge(smoothed[merge_cols], on="createdTime", how="left")

        observed = res["hrvValue"].notna()
        res["gap_flag"] = (~observed).astype(int)

        out_cols = ["createdTime", "hrvValue", "minute_diff", "smoothed_hrv",
                    "gap_flag", "chunk_id"]
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        res[out_cols].to_csv(Path(output_dir) / f"{stem}_krlst.csv", index=False)

        n_smoothed = int(res["smoothed_hrv"].notna().sum())
        n_chunks = int(res.loc[observed, "chunk_id"].nunique())
        if n_smoothed == 0:
            return {"file": file_path.name, "status": "failed",
                    "reason": "No chunk long enough to smooth"}

        # Positivity guard (H1): every smoothed value must be strictly > 0.
        mn = float(res["smoothed_hrv"].min(skipna=True))
        if not (mn > 0):
            return {"file": file_path.name, "status": "failed",
                    "reason": f"Non-positive smoothed value ({mn})"}

        return {"file": file_path.name, "status": "success", "n_rows": len(res),
                "n_smoothed": n_smoothed, "n_chunks": n_chunks,
                "min_smoothed": round(mn, 3)}

    except Exception as e:
        return {"file": file_path.name, "status": "failed", "reason": str(e)}


def run_krlst_smoothing(budget=GLOBAL_BUDGET, lam=GLOBAL_LAMBDA):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(INPUT_DIR.glob("*_processed.csv"))
    if not csv_files:
        print(f"ERROR: No processed CSVs found in {INPUT_DIR.resolve()}")
        return

    print("\n--- KRLST SMOOTHER (kernel recursive least-squares tracker) ---")
    print(f"Input:      {INPUT_DIR.resolve()}")
    print(f"Output:     {OUTPUT_DIR.resolve()}")
    print(f"Patients:   {len(csv_files)}")
    print(f"Budget M={budget}  lambda={lam}  Gap threshold=180 min")

    max_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Workers:    {max_workers}\n")

    args_list = [(f, OUTPUT_DIR, budget, lam) for f in csv_files]
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_patient, a): a for a in args_list}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(args_list), desc="KRLST smoothing"):
            results.append(future.result())

    df_log = pd.DataFrame(results)
    log_path = RESULTS_DIR / "smoothing_krlst_log.csv"
    df_log.to_csv(log_path, index=False)

    successes = df_log[df_log["status"] == "success"]
    failures = df_log[df_log["status"] == "failed"]

    print("\n" + "=" * 60)
    print("KRLST SMOOTHING COMPLETE")
    print("=" * 60)
    print(f"Success: {len(successes)} / {len(csv_files)}")
    print(f"Failed:  {len(failures)}")
    print(f"Output:  {OUTPUT_DIR.resolve()}")
    print(f"Log:     {log_path.resolve()}")

    if len(failures) > 0:
        print("\nFailures:")
        for _, row in failures.iterrows():
            print(f"  - {row['file']}: {row['reason']}")


if __name__ == "__main__":
    run_krlst_smoothing()
