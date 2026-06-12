from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

RAW_DATA_DIR = Path("./raw_data")
OUTPUT_DIR = Path("./data/processed")
PATIENT_LIST = RAW_DATA_DIR / "processed_users.txt"
GAP_THRESHOLD_MIN = 180
FILL_INTERVAL_MIN = 10

def _load_hrv_csvs(patient_dir: Path) -> pd.DataFrame:
    """Loads and concatenates valid HRV CSVs, extracting only required columns."""
    csv_paths = [p for p in patient_dir.glob("hrv/*.csv") if not p.name.startswith((".", "_"))]
    if not csv_paths:
        return pd.DataFrame()

    frames = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
            df.rename(columns=lambda x: x.strip(), inplace=True)
            if {"createdTime", "hrvValue"}.issubset(df.columns):
                frames.append(df[["createdTime", "hrvValue"]])
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            continue

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def _insert_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized insertion of NaN rows at FILL_INTERVAL_MIN for gaps > GAP_THRESHOLD_MIN."""
    df["createdTime"] = pd.to_datetime(df["createdTime"], errors="coerce")
    df["hrvValue"] = pd.to_numeric(df["hrvValue"], errors="coerce")
    df = df.dropna(subset=["createdTime"]).sort_values("createdTime").reset_index(drop=True)

    if df.empty:
        return pd.DataFrame(columns=["createdTime", "hrvValue", "minute_diff"])

    diff_mins = df["createdTime"].diff().dt.total_seconds() / 60.0
    gap_mask = diff_mins > GAP_THRESHOLD_MIN

    fill_frames = []
    for idx in df[gap_mask].index:
        fill_times = pd.date_range(
            start=df.at[idx - 1, "createdTime"] + pd.Timedelta(minutes=FILL_INTERVAL_MIN),
            end=df.at[idx, "createdTime"],
            freq=f"{FILL_INTERVAL_MIN}min",
            inclusive="left"
        )
        fill_frames.append(pd.DataFrame({"createdTime": fill_times, "hrvValue": np.nan}))

    if fill_frames:
        df = pd.concat([df, *fill_frames], ignore_index=True).sort_values("createdTime").reset_index(drop=True)

    df["minute_diff"] = df["createdTime"].diff().dt.total_seconds() / 60.0
    return df[["createdTime", "hrvValue", "minute_diff"]]

def run_preprocessing() -> None:
    """Executes the preprocessing pipeline across all patient directories."""
    if not PATIENT_LIST.exists():
        return print(f"ERROR: Patient list not found at {PATIENT_LIST}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (results_dir := Path("./data/results")).mkdir(parents=True, exist_ok=True)

    pids = [line.strip().strip('"') for line in PATIENT_LIST.read_text().splitlines() if line.strip()]
    results = []

    for pid in tqdm(pids, desc="Preprocessing HRV Time-Series"):
        target = RAW_DATA_DIR / f'"{pid}"'
        patient_dir = target if target.exists() else RAW_DATA_DIR / pid

        if not patient_dir.exists():
            results.append((pid, "failed", "Patient directory not found"))
            continue

        if (df_raw := _load_hrv_csvs(patient_dir)).empty:
            results.append((pid, "failed", "No HRV CSVs found"))
            continue

        if (df_out := _insert_nan_rows(df_raw)).empty:
            results.append((pid, "failed", "No valid rows after processing"))
            continue

        df_out.to_csv(OUTPUT_DIR / f"{pid}_processed.csv", index=False)
        real = df_out["hrvValue"].notna().sum()
        results.append((pid, "success", f"rows={len(df_out)} real={real} nan={len(df_out) - real}"))

    df_log = pd.DataFrame(results, columns=["patient", "status", "detail"])
    df_log.to_csv(log_path := results_dir / "phase1_preprocess_log.csv", index=False)

    fails = df_log[df_log["status"] == "failed"]
    print(f"\n{'=' * 60}\nPREPROCESSING COMPLETE\n{'=' * 60}")
    print(f"Processed: {len(df_log) - len(fails)} / {len(df_log)}\nFailed:    {len(fails)}")
    print(f"Output:    {OUTPUT_DIR.resolve()}\nLog:       {log_path.resolve()}")
    
    if not fails.empty:
        print("\nFirst 5 failures:\n" + "\n".join(f"  - {r.patient}: {r.detail}" for _, r in fails.head(5).iterrows()))

if __name__ == "__main__":
    run_preprocessing()