"""Stage 2h (alternative smoother): Log-MHW (log multiplicative Holt-Winters).

A NEW, independent smoothing method for benchmarking against the SMC+FFBS
particle filter (`02_run_filters.py`), RS-PF (`02b`), KRLST (`02c`), GP-SSM
(`02d`), OSSA (`02e`), Kim (`02f`) and Gamma DGLM (`02g`). It does NOT modify or
read their outputs.

  Input :  data/processed/<pid>_processed.csv   (from 00_preprocess_raw.py)
  Output:  data/smoothed_logmhw/<pid>_logmhw.csv

Output columns:
  createdTime, hrvValue, minute_diff, smoothed_hrv, gap_flag, chunk_id

  * smoothed_hrv = exp(level + season) from causal log Holt-Winters (> 0)
  * chunk_id     = H8 segment id (resets at >= 180-min gaps)

Run from the asthma-prediction/ directory:
    python src/02h_run_logmhw.py
"""

import warnings
from pathlib import Path
import concurrent.futures
import multiprocessing

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

try:
    from logmhw_smoother import smooth_dataframe, LogMHWConfig
except ImportError:  # pragma: no cover
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from logmhw_smoother import smooth_dataframe, LogMHWConfig

INPUT_DIR = Path("./data/processed")
OUTPUT_DIR = Path("./data/smoothed_logmhw")
RESULTS_DIR = Path("./data/results")

# ---- Method configuration ------------------------------------------------
GLOBAL_ALPHA = 0.08      # level smoothing (smaller => smoother)
GLOBAL_GAMMA = 0.02      # seasonal smoothing
GLOBAL_OPTIMIZE = True   # optimise alpha/beta/phi per patient via statsmodels


def process_patient(args):
    file_path, output_dir, alpha, gamma, optimize = args
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()

        if not {"createdTime", "hrvValue", "minute_diff"}.issubset(df.columns):
            return {"file": file_path.name, "status": "failed",
                    "reason": f"Missing columns. Found: {list(df.columns)}"}

        stem = file_path.stem.replace("_processed", "")
        df["createdTime"] = pd.to_datetime(df["createdTime"])

        # H8: segment on observed readings only (the 10-min NaN fillers across
        # >180-min gaps would otherwise hide the real voids from the chunker).
        obs = df[df["hrvValue"].notna()].copy()
        obs["patient_id"] = stem

        cfg = LogMHWConfig(alpha=alpha, gamma=gamma, optimize=optimize)
        smoothed = smooth_dataframe(
            obs, patient_col="patient_id", timestamp_col="createdTime",
            value_col="hrvValue", config=cfg, out_col="smoothed_hrv",
        )

        res = df.copy()
        merge_cols = ["createdTime", "smoothed_hrv", "chunk_id"]
        res = res.merge(smoothed[merge_cols], on="createdTime", how="left")

        observed = res["hrvValue"].notna()
        res["gap_flag"] = (~observed).astype(int)
        # Track B − Track A: the residual the CPD ensemble runs on.
        res["residual_hrv"] = res["hrvValue"] - res["smoothed_hrv"]

        out_cols = ["createdTime", "hrvValue", "minute_diff", "smoothed_hrv",
                    "residual_hrv", "gap_flag", "chunk_id"]
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        res[out_cols].to_csv(Path(output_dir) / f"{stem}_logmhw.csv", index=False)

        n_smoothed = int(res["smoothed_hrv"].notna().sum())
        n_chunks = int(res.loc[observed, "chunk_id"].nunique())
        if n_smoothed == 0:
            return {"file": file_path.name, "status": "failed",
                    "reason": "No chunk long enough to smooth"}

        mn = float(res["smoothed_hrv"].min(skipna=True))
        if not (mn > 0):
            return {"file": file_path.name, "status": "failed",
                    "reason": f"Non-positive smoothed value ({mn})"}

        return {"file": file_path.name, "status": "success", "n_rows": len(res),
                "n_smoothed": n_smoothed, "n_chunks": n_chunks,
                "min_smoothed": round(mn, 3)}

    except Exception as e:
        return {"file": file_path.name, "status": "failed", "reason": str(e)}


def run_logmhw_smoothing(alpha=GLOBAL_ALPHA, gamma=GLOBAL_GAMMA, optimize=GLOBAL_OPTIMIZE):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(INPUT_DIR.glob("*_processed.csv"))
    if not csv_files:
        print(f"ERROR: No processed CSVs found in {INPUT_DIR.resolve()}")
        return

    print("\n--- LOG-MHW SMOOTHER (log multiplicative Holt-Winters) ---")
    print(f"Input:      {INPUT_DIR.resolve()}")
    print(f"Output:     {OUTPUT_DIR.resolve()}")
    print(f"Patients:   {len(csv_files)}")
    print(f"alpha={alpha}  gamma={gamma}  optimize={optimize}  Gap threshold=180 min")

    max_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Workers:    {max_workers}\n")

    args_list = [(f, OUTPUT_DIR, alpha, gamma, optimize) for f in csv_files]
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_patient, a): a for a in args_list}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(args_list), desc="Log-MHW smoothing"):
            results.append(future.result())

    df_log = pd.DataFrame(results)
    log_path = RESULTS_DIR / "smoothing_logmhw_log.csv"
    df_log.to_csv(log_path, index=False)

    successes = df_log[df_log["status"] == "success"]
    failures = df_log[df_log["status"] == "failed"]

    print("\n" + "=" * 60)
    print("LOG-MHW SMOOTHING COMPLETE")
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
    run_logmhw_smoothing()
