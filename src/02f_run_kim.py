"""Stage 2f (alternative smoother): Markov-Switching State-Space (Kim filter).

A NEW, independent smoothing method for benchmarking against the SMC+FFBS
particle filter (`02_run_filters.py`), RS-PF (`02b`), KRLST (`02c`), GP-SSM
(`02d`) and OSSA (`02e`). It does NOT modify or read their outputs.

  Input :  data/processed/<pid>_processed.csv   (from 00_preprocess_raw.py)
  Output:  data/smoothed_kim/<pid>_kim.csv

Output columns:
  createdTime, hrvValue, minute_diff, smoothed_hrv, gap_flag, chunk_id

  * smoothed_hrv = Kim-filter regime-mixed filtered level (single output, > 0)
  * chunk_id     = H8 segment id (resets at >= 180-min gaps)

Run from the asthma-prediction/ directory:
    python src/02f_run_kim.py
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
    from kim_smoother import smooth_dataframe, KimConfig
except ImportError:  # pragma: no cover
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from kim_smoother import smooth_dataframe, KimConfig

INPUT_DIR = Path("./data/processed")
OUTPUT_DIR = Path("./data/smoothed_kim")
RESULTS_DIR = Path("./data/results")

# ---- Method configuration ------------------------------------------------
GLOBAL_Q_SCALE = 0.1     # process-noise multiplier (smaller => smoother)
GLOBAL_R_SCALE = 3.0     # obs-noise multiplier    (larger  => smoother)


def process_patient(args):
    """Smooth one processed patient CSV with the Kim filter and write the output."""
    file_path, output_dir, q_scale, r_scale = args
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()

        if not {"createdTime", "hrvValue", "minute_diff"}.issubset(df.columns):
            return {"file": file_path.name, "status": "failed",
                    "reason": f"Missing columns. Found: {list(df.columns)}"}

        stem = file_path.stem.replace("_processed", "")
        df["createdTime"] = pd.to_datetime(df["createdTime"])

        # CRITICAL (H8): 00_preprocess_raw.py inserts NaN-hrv filler rows every
        # 10 min across gaps > 180 min, which would HIDE the real voids from the
        # chunker. The H8 rule is on consecutive *readings*, so segment on
        # observed rows only, then merge results back onto the full grid.
        obs = df[df["hrvValue"].notna()].copy()
        obs["patient_id"] = stem

        cfg = KimConfig(q_scale=q_scale, r_scale=r_scale)
        smoothed = smooth_dataframe(
            obs, patient_col="patient_id", timestamp_col="createdTime",
            value_col="hrvValue", config=cfg, out_col="smoothed_hrv",
        )

        res = df.copy()
        merge_cols = ["createdTime", "smoothed_hrv", "chunk_id"]
        res = res.merge(smoothed[merge_cols], on="createdTime", how="left")

        observed = res["hrvValue"].notna()
        res["gap_flag"] = (~observed).astype(int)

        out_cols = ["createdTime", "hrvValue", "minute_diff", "smoothed_hrv",
                    "gap_flag", "chunk_id"]
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        res[out_cols].to_csv(Path(output_dir) / f"{stem}_kim.csv", index=False)

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


def run_kim_smoothing(q_scale=GLOBAL_Q_SCALE, r_scale=GLOBAL_R_SCALE):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(INPUT_DIR.glob("*_processed.csv"))
    if not csv_files:
        print(f"ERROR: No processed CSVs found in {INPUT_DIR.resolve()}")
        return

    print("\n--- KIM FILTER SMOOTHER (Markov-switching SSM: Hamilton + Kalman) ---")
    print(f"Input:      {INPUT_DIR.resolve()}")
    print(f"Output:     {OUTPUT_DIR.resolve()}")
    print(f"Patients:   {len(csv_files)}")
    print(f"q_scale={q_scale}  r_scale={r_scale}  Regimes=2  Gap threshold=180 min")

    max_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Workers:    {max_workers}\n")

    args_list = [(f, OUTPUT_DIR, q_scale, r_scale) for f in csv_files]
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_patient, a): a for a in args_list}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(args_list), desc="Kim smoothing"):
            results.append(future.result())

    df_log = pd.DataFrame(results)
    log_path = RESULTS_DIR / "smoothing_kim_log.csv"
    df_log.to_csv(log_path, index=False)

    successes = df_log[df_log["status"] == "success"]
    failures = df_log[df_log["status"] == "failed"]

    print("\n" + "=" * 60)
    print("KIM FILTER SMOOTHING COMPLETE")
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
    run_kim_smoothing()
