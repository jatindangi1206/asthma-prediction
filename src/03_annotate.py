import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent))

import bocpd
import hmm_cpd
import kalman_cpd
import kcp

INPUT_DIR  = Path("./data/smoothed")
OUTPUT_DIR = Path("./data/annotated")
RESULTS_DIR = Path("./data/results")

METHODS = [
    ("bocpd",  bocpd),
    ("kcp",    kcp),
    ("kalman", kalman_cpd),
    ("hmm",    hmm_cpd),
]


def annotate_patient(file_path: Path) -> dict:
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()

        required = {"createdTime", "smoothed_hrv"}
        if not required.issubset(df.columns):
            return {"file": file_path.name, "status": "failed",
                    "reason": f"Missing columns. Found: {list(df.columns)}"}

        df["createdTime"] = pd.to_datetime(df["createdTime"])
        df = df.sort_values("createdTime").reset_index(drop=True)

        values = df["smoothed_hrv"].values.astype(float)

        # Elapsed minutes from start — used as the time axis for Kalman CPD
        elapsed_min = (
            (df["createdTime"] - df["createdTime"].iloc[0])
            .dt.total_seconds() / 60.0
        ).values

        for method_name, module in METHODS:
            types, degrees = module.detect(elapsed_min, values)
            df[f"{method_name}_type"]   = types
            df[f"{method_name}_degree"] = degrees

        stem = file_path.stem.replace("_smoothed", "")
        out_file = OUTPUT_DIR / f"{stem}_annotated.csv"
        df.to_csv(out_file, index=False)

        return {"file": file_path.name, "status": "success", "n_rows": len(df)}

    except Exception as e:
        return {"file": file_path.name, "status": "failed", "reason": str(e)}


def run_annotation() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(INPUT_DIR.glob("*_smoothed.csv"))
    if not csv_files:
        print(f"ERROR: No smoothed CSVs found in {INPUT_DIR.resolve()}")
        return

    print(f"\n--- CPD ANNOTATION ---")
    print(f"Input:    {INPUT_DIR.resolve()}")
    print(f"Output:   {OUTPUT_DIR.resolve()}")
    print(f"Patients: {len(csv_files)}")
    print(f"Methods:  {[m for m, _ in METHODS]}\n")

    results = []
    for f in tqdm(csv_files, desc="Annotating"):
        results.append(annotate_patient(f))

    df_log = pd.DataFrame(results)
    log_path = RESULTS_DIR / "annotation_log.csv"
    df_log.to_csv(log_path, index=False)

    successes = df_log[df_log["status"] == "success"]
    failures  = df_log[df_log["status"] == "failed"]

    print("\n" + "=" * 60)
    print("ANNOTATION COMPLETE")
    print("=" * 60)
    print(f"Success: {len(successes)} / {len(csv_files)}")
    print(f"Failed:  {len(failures)}")
    print(f"Output:  {OUTPUT_DIR.resolve()}")
    print(f"Log:     {log_path.resolve()}")

    if len(failures) > 0:
        print("\nFailures:")
        for _, row in failures.iterrows():
            print(f"  - {row['file']}: {row['reason']}")

    # Print column layout of a sample output
    sample = sorted(OUTPUT_DIR.glob("*_annotated.csv"))
    if sample:
        cols = pd.read_csv(sample[0], nrows=0).columns.tolist()
        print(f"\nOutput columns: {cols}")


if __name__ == "__main__":
    run_annotation()
