"""Stage 1 entrypoint — align one patient onto the HRV grid.

Thin wrapper over ``ingest_wearable`` (the alignment engine) that uses the
central config paths and writes ``data/processed/<patient>_aligned.csv``.

    python -m pipeline.preprocessing.preprocessing <patient_id>
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline import config
from pipeline.preprocessing.ingest_wearable import ingest_patient, _resolve_patient_dir


def preprocess_patient(patient_id: str) -> Path:
    """Run Stage 1; return the path to ``<patient>_aligned.csv``."""
    config.ensure_dirs()
    patient_dir = _resolve_patient_dir(config.RAW_DATA_DIR, patient_id)
    print(f"=== Stage 1: preprocessing {patient_dir.name} ===")
    out_csv, _, report = ingest_patient(patient_dir, config.PROCESSED_DIR)
    print("\n" + report)
    print(f"\n[{patient_id}] aligned -> {out_csv}")
    return out_csv


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m pipeline.preprocessing.preprocessing <patient_id>")
        raise SystemExit(2)
    preprocess_patient(sys.argv[1])
