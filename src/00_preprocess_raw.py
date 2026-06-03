import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

RAW_DATA_DIR = Path("./raw_data")
OUTPUT_DIR = Path("./data/processed")
PATIENT_LIST = RAW_DATA_DIR / "processed_users.txt"
GAP_THRESHOLD_MIN = 180
FILL_INTERVAL_MIN = 10


def _load_hrv_csvs(patient_dir: Path) -> pd.DataFrame:
    hrv_dir = patient_dir / "hrv"
    if not hrv_dir.exists():
        return pd.DataFrame()

    csv_files = [
        p for p in hrv_dir.glob("*.csv")
        if not p.name.startswith(".") and not p.name.endswith(".crc") and not p.name.startswith("_")
    ]
    if not csv_files:
        return pd.DataFrame()

    frames = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            df.columns = df.columns.str.strip()
            if "createdTime" in df.columns and "hrvValue" in df.columns:
                frames.append(df[["createdTime", "hrvValue"]])
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _insert_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every consecutive pair of rows whose gap exceeds GAP_THRESHOLD_MIN,
    insert placeholder rows at FILL_INTERVAL_MIN intervals.
    Inserted rows have NaN for hrvValue and minute_diff.
    """
    df = df.copy()
    df["createdTime"] = pd.to_datetime(df["createdTime"], errors="coerce")
    df["hrvValue"] = pd.to_numeric(df["hrvValue"], errors="coerce")
    df = df.dropna(subset=["createdTime"]).sort_values("createdTime").reset_index(drop=True)

    if len(df) == 0:
        return pd.DataFrame(columns=["createdTime", "hrvValue", "minute_diff"])

    output_rows = []

    for i, row in df.iterrows():
        if i == 0:
            output_rows.append({"createdTime": row["createdTime"], "hrvValue": row["hrvValue"]})
            continue

        prev_time = df.loc[i - 1, "createdTime"]
        curr_time = row["createdTime"]
        gap_min = (curr_time - prev_time).total_seconds() / 60.0

        if gap_min > GAP_THRESHOLD_MIN:
            # Insert NaN placeholder rows every FILL_INTERVAL_MIN minutes
            fill_time = prev_time + pd.Timedelta(minutes=FILL_INTERVAL_MIN)
            while fill_time < curr_time:
                output_rows.append({"createdTime": fill_time, "hrvValue": np.nan})
                fill_time += pd.Timedelta(minutes=FILL_INTERVAL_MIN)

        output_rows.append({"createdTime": curr_time, "hrvValue": row["hrvValue"]})

    result = pd.DataFrame(output_rows)
    result = result.sort_values("createdTime").reset_index(drop=True)

    # Recalculate minute_diff on the final frame (NaN for first row)
    result["minute_diff"] = (
        result["createdTime"].diff().dt.total_seconds() / 60.0
    )

    return result[["createdTime", "hrvValue", "minute_diff"]]


def run_preprocessing() -> None:
    if not PATIENT_LIST.exists():
        print(f"ERROR: Patient list not found at {PATIENT_LIST}")
        return

    patient_ids = [
        line.strip().strip('"')
        for line in PATIENT_LIST.read_text().splitlines()
        if line.strip()
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_dir = Path("./data/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(patient_ids)} patients in {PATIENT_LIST}")

    results = []
    for pid in tqdm(patient_ids, desc="Preprocessing HRV"):
        # Try quoted folder name first (e.g. "0010"), then bare name
        patient_dir = RAW_DATA_DIR / f'"{pid}"'
        if not patient_dir.exists():
            patient_dir = RAW_DATA_DIR / pid
        if not patient_dir.exists():
            results.append((pid, "failed", "Patient directory not found"))
            continue

        df_raw = _load_hrv_csvs(patient_dir)
        if df_raw.empty:
            results.append((pid, "failed", "No HRV CSVs found"))
            continue

        df_out = _insert_nan_rows(df_raw)
        if df_out.empty:
            results.append((pid, "failed", "No valid rows after processing"))
            continue

        out_path = OUTPUT_DIR / f"{pid}_processed.csv"
        df_out.to_csv(out_path, index=False)
        n_real = int(df_out["hrvValue"].notna().sum())
        n_nan = int(df_out["hrvValue"].isna().sum())
        results.append((pid, "success", f"rows={len(df_out)} real={n_real} nan_inserted={n_nan}"))

    df_log = pd.DataFrame(results, columns=["patient", "status", "detail"])
    log_path = results_dir / "phase1_preprocess_log.csv"
    df_log.to_csv(log_path, index=False)

    successes = df_log[df_log["status"] == "success"]
    failures = df_log[df_log["status"] == "failed"]

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Processed: {len(successes)} / {len(df_log)}")
    print(f"Failed:    {len(failures)}")
    print(f"Output:    {OUTPUT_DIR.resolve()}")
    print(f"Log:       {log_path.resolve()}")

    if len(failures) > 0:
        print("\nFirst 5 failures:")
        for _, row in failures.head(5).iterrows():
            print(f"  - {row['patient']}: {row['detail']}")


if __name__ == "__main__":
    run_preprocessing()
