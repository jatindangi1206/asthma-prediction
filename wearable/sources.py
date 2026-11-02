"""Data sources: bring the physiological signals in from whatever the export looks like.

Each source hides one export format behind two functions, so the ingest stage — and
therefore everything downstream — is identical whatever the input was:

    patients()   -> [pid, ...]          which patients this source offers
    load(pid)    -> PatientData         that patient's readings, and labels if it has any

`PatientData.readings` carries `createdTime` plus one column per channel the export
actually holds, named as `config.CHANNELS` says — unsorted and ungapped, since ingest
applies the shared H8 gap rule per channel. A source that only has HRV returns only
`hrvValue`, and nothing downstream cares. `PatientData.labels` is optional day-level
ground truth and is `None` for sources that have none.

Adding a format (a new device, a new study export) is one `patients`/`load` pair plus one
`Source(...)` line at the bottom. Nothing else in the pipeline changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from wearable.config import CHANNELS, RAW_DIR

PASS1_DIR = RAW_DIR / "pass1"
SPARK_DIR = RAW_DIR / "spark"

# Day-level ground truth; see data/raw/pass1/README.md §3. These repeat on
# every minute of a day, so they are collapsed to one row per patient-day.
LABEL_COLS = ["exac_day", "episode", "sample", "label",
              "onset_valid", "onset_label", "onset_id"]
_STR_COLS = {"episode": str, "onset_id": str, "p_id": str}


@dataclass
class PatientData:
    pid: str
    readings: pd.DataFrame            # createdTime + one column per available channel
    labels: pd.DataFrame | None = None  # p_id, date, <LABEL_COLS>
    note: str = ""                    # why readings is empty, when it is

    def channels(self) -> list[str]:
        """The channels this patient actually has at least one reading of."""
        return [c for c, col in CHANNELS.items()
                if col in self.readings.columns
                and pd.to_numeric(self.readings[col], errors="coerce").notna().any()]


# ---------------------------------------------------------------------------
# pass1 — the labelled minute-grid cohort (AIIMS a/p + COPD)
# ---------------------------------------------------------------------------
def _pass1_index() -> pd.DataFrame:
    """The 221 `in_models` patients: AIIMS adults and children and COPD with readable
    exacerbation dates. ACHE has no exacerbation record, and four patients (c005, c032,
    c037, p025) have unreadable dates, so none of them are in the models."""
    idx = pd.read_csv(PASS1_DIR / "patients.csv", dtype=_STR_COLS)
    return idx[idx["in_models"] == 1]


def pass1_patients() -> list[str]:
    return _pass1_index()["p_id"].tolist()


def pass1_load(pid: str) -> PatientData:
    row = _pass1_index().set_index("p_id").loc[pid]
    df = pd.read_csv(PASS1_DIR / row["file"], parse_dates=["time"], dtype=_STR_COLS)
    if df.empty:
        return PatientData(pid, pd.DataFrame(columns=["createdTime", "hrvValue"]),
                           note="empty file")

    # Collapse labels BEFORE filtering on HRV, so an exacerbation day that happens to
    # carry no HRV reading is still on record.
    labels = (df.assign(date=df["time"].dt.normalize())
                .groupby("date", as_index=False)[LABEL_COLS].first())
    labels.insert(0, "p_id", pid)
    # Which channels each day actually holds a reading of. The study's own `sample`
    # column means "the day has at least one HRV reading" (raw README §3), which bakes
    # an HRV-only pipeline into the ground truth — a day with 710 heart-rate readings
    # and no HRV is marked unscoreable. Storing the FACTS here lets `join` choose the
    # rule (`join.VALIDITY`) without re-ingesting, and lets the wider rule be measured
    # against the study's own before anything is changed.
    day = df.assign(date=df["time"].dt.normalize()).groupby("date")
    for ch, col in CHANNELS.items():
        raw_name = {"hrv": "hrv"}.get(ch, ch)
        labels[f"has_{ch}"] = (
            day[raw_name].apply(lambda v: int(v.notna().any())).reindex(
                labels["date"]).fillna(0).astype(int).to_numpy()
            if raw_name in df.columns else 0)
    # The cohort group (aiims_adult / aiims_paed / copd) lives in the patient index, not
    # in the minute file. Scores are reported per group and the split is stratified by
    # it, so it travels with the labels.
    labels["group"] = row["group"]

    # Every channel the export holds, on the raw timestamps. Rows are NOT filtered on
    # any one channel: ingest builds each channel's own grid from its own observations,
    # so keeping a row that has heart rate but no HRV is what lets the 30 patients with
    # no surviving HRV into the pipeline at all.
    have = {col: raw for ch, col in CHANNELS.items()
            if (raw := {"hrv": "hrv"}.get(ch, ch)) in df.columns}
    readings = df[["time", *have.values()]].rename(
        columns={"time": "createdTime", **{raw: col for col, raw in have.items()}})
    present = [c for c, col in CHANNELS.items()
               if col in readings.columns and readings[col].notna().any()]
    note = "" if present else "no readings in any channel"
    return PatientData(pid, readings, labels, note)


# ---------------------------------------------------------------------------
# spark — the older per-vital Spark-partition export (unlabelled)
# ---------------------------------------------------------------------------
def spark_patients() -> list[str]:
    listing = SPARK_DIR / "processed_users.txt"
    if not listing.exists():
        return []
    return [l.strip().strip('"') for l in listing.read_text().splitlines() if l.strip()]


def spark_load(pid: str) -> PatientData:
    # Raw folder names are sometimes quoted on disk (`"0010"`); accept either.
    quoted = SPARK_DIR / f'"{pid}"'
    patient_dir = quoted if quoted.exists() else SPARK_DIR / pid
    empty = pd.DataFrame(columns=["createdTime", "hrvValue"])
    if not patient_dir.exists():
        return PatientData(pid, empty, note="patient directory not found")

    frames = []
    for path in patient_dir.glob("hrv/*.csv"):
        if path.name.startswith((".", "_")):
            continue
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            continue
        df.rename(columns=lambda c: c.strip(), inplace=True)
        if {"createdTime", "hrvValue"}.issubset(df.columns):
            frames.append(df[["createdTime", "hrvValue"]])

    if not frames:
        return PatientData(pid, empty, note="no HRV CSVs found")
    return PatientData(pid, pd.concat(frames, ignore_index=True))


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Source:
    key: str
    label: str
    patients: Callable[[], list[str]]
    load: Callable[[str], PatientData]
    has_labels: bool


REGISTRY: dict[str, Source] = {
    s.key: s for s in [
        Source("pass1", "pass1 minute grid (221 labelled patients)",
               pass1_patients, pass1_load, has_labels=True),
        Source("spark", "spark export (109 patients, unlabelled)",
               spark_patients, spark_load, has_labels=False),
    ]
}

DEFAULT_SOURCE = "pass1"


def get(key: str) -> Source:
    if key not in REGISTRY:
        raise KeyError(f"unknown source {key!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[key]
