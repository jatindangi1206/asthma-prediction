"""Orchestrate the four pipeline stages for one patient, in order.

Each stage reads the previous stage's output:
    1. preprocessing -> data/processed/<id>_aligned.csv
    2. cleaning      -> data/processed/<id>_cleaned.csv
    3. smoothing     -> data/processed/<id>_smoothed.csv
    4. cpd           -> pipeline/outputs/<id>_final.csv

    python pipeline/run_pipeline.py <patient_id>
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.preprocessing.preprocessing import preprocess_patient
from pipeline.cleaning.cleaning import clean_patient
from pipeline.smoothing.smoothing import smooth_patient
from pipeline.cpd.cpd_pipeline import run_cpd


def run_pipeline(patient_id: str) -> Path:
    preprocess_patient(patient_id)
    clean_patient(patient_id)
    smooth_patient(patient_id)
    final = run_cpd(patient_id)
    print(f"\n=== Pipeline complete for {patient_id} ===\n{final}")
    return final


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python pipeline/run_pipeline.py <patient_id>")
        raise SystemExit(2)
    run_pipeline(sys.argv[1])
