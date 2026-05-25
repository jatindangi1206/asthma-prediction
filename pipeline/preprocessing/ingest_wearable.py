"""Fuse six wearable signals for one patient into a single HRV-anchored timeline.

Real data layout (AIIMS Spark exports):
    pipeline/raw_data/aiims-exports/<patient>/<signal>/part-*.csv
        + _SUCCESS and .crc sidecars (ignored)

The six signals and their columns (folder name -> columns):
    heartrate   logDateTime, lastRate
    hrv         createdTime, hrvValue          <-- PRIMARY / base grid
    temperature createdTime, temperature       (degF; < 90 => sensor off-skin)
    steps       logDateTime, logEndTime, steps, distance
    sleep       logDateTime, logEndTime, description  (description IGNORED)
    spo2        createdTime, spo2Value         (very sparse)

Output: one CSV per patient on the HRV timeline, with HRV as `hrvValue` (the only
signal the downstream pipeline smooths / change-point-detects) plus context columns
carried alongside. The output obeys the contract Pipeline A's later stages expect:
later stages append columns and never reorder time, so context rides through to the
final annotated CSV (run_cpd.py is patched separately to carry unknown columns).

Alignment (approved design):
  * Base grid  = the patient's HRV reading timestamps (irregular ~10 min). One output
    row per HRV reading. `minute` = round((ts - first_hrv_ts) / 60), integer.
    Bin i spans [hrv_ts_i, hrv_ts_{i+1}); bin widths vary (HRV is irregular).
  * heartrate  = mean of lastRate per bin            (dense -> aggregate)
  * steps      = sum of steps per bin (assigned by logDateTime)  (dense -> aggregate)
  * temperature= nearest reading within --temp-tol min (merge_asof, no fill beyond)
  * spo2       = nearest reading within --spo2-tol min (merge_asof, no fill beyond)
  * is_asleep  = 1 if the HRV timestamp falls inside ANY [logDateTime, logEndTime]
                 sleep interval (overlaps allowed); description codes are NOT decoded.

Masks: hr_mask / steps_mask / temp_mask / spo2_mask are 1 where a real reading
contributed, else 0. watch_off_flag = 1 where temperature < 90 (raw temp kept).
is_asleep is always defined, so it has no mask.

Usage:
    python pipeline/preprocessing/ingest_wearable.py --patient a001 --days 3
    python pipeline/preprocessing/ingest_wearable.py --all
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # allow running as a bare script as well as -m
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline import config


# Paths are centralised in config.py (agreed layout):
#   data/raw_data/<patient>/ in, data/processed/<patient>_aligned.csv out.
EXPORTS_DIR = config.RAW_DATA_DIR
OUTPUT_DIR = config.PROCESSED_DIR

OFF_SKIN_TEMP = config.WATCH_OFF_TEMP_THRESHOLD  # degF; below this => watch off wrist
DEFAULT_TEMP_TOL_MIN = 20.0  # temperature cadence ~15 min
DEFAULT_SPO2_TOL_MIN = 30.0  # spo2 is extremely sparse


@dataclass
class SignalSpec:
	name: str
	folders: list[str]  # candidate folder names (real layout uses the first)
	time_cols: list[str]
	value_cols: list[str] = field(default_factory=list)
	end_cols: list[str] = field(default_factory=list)


# `realtimetemphr` (present for 2 patients) is a different combined export and is
# intentionally not mapped here; it is reported as skipped.
SIGNALS = {
	"hrv": SignalSpec("hrv", ["hrv"], ["createdTime", "logDateTime"], ["hrvValue"]),
	"heartrate": SignalSpec("heartrate", ["heartrate", "heart_rate"], ["logDateTime", "createdTime"], ["lastRate"]),
	"temperature": SignalSpec("temperature", ["temperature", "temp"], ["createdTime", "logDateTime"], ["temperature"]),
	"steps": SignalSpec("steps", ["steps"], ["logDateTime"], ["steps"], ["logEndTime"]),
	"sleep": SignalSpec("sleep", ["sleep"], ["logDateTime"], [], ["logEndTime"]),
	"spo2": SignalSpec("spo2", ["spo2", "SpO2"], ["createdTime", "logDateTime"], ["spo2Value", "spo2"]),
}


def _resolve_column(columns: pd.Index, candidates: Iterable[str], label: str) -> str:
	for name in candidates:
		if name in columns:
			return name
	raise ValueError(f"Missing {label} column. Expected one of: {', '.join(candidates)}")


def _parse_datetime(series: pd.Series) -> pd.Series:
	return pd.to_datetime(series, errors="coerce")


def _find_signal_dir(patient_dir: Path, candidates: list[str]) -> Path | None:
	for name in candidates:
		candidate = patient_dir / name
		if candidate.is_dir():
			return candidate
	return None


def _load_part_csv(signal_dir: Path) -> tuple[pd.DataFrame | None, str | None]:
	"""Load the Spark part-*.csv from a signal folder, ignoring _SUCCESS / .crc."""
	parts = sorted(p for p in signal_dir.glob("part-*.csv") if not p.name.startswith("."))
	if not parts:
		return None, None
	return pd.read_csv(parts[0]), parts[0].name


def load_signal(patient_dir: Path, spec: SignalSpec) -> tuple[pd.DataFrame | None, dict]:
	"""Load one signal as a time-sorted frame. Returns (df, info)."""
	info = {"loaded": False, "folder": None, "file": None, "rows": 0}
	signal_dir = _find_signal_dir(patient_dir, spec.folders)
	if signal_dir is None:
		return None, info

	df, fname = _load_part_csv(signal_dir)
	info["folder"] = signal_dir.name
	if df is None or df.empty:
		return None, info

	time_col = _resolve_column(df.columns, spec.time_cols, f"{spec.name} time")
	df["_t"] = _parse_datetime(df[time_col])
	if spec.end_cols:
		end_col = _resolve_column(df.columns, spec.end_cols, f"{spec.name} end")
		df["_t_end"] = _parse_datetime(df[end_col])

	df = df.dropna(subset=["_t"]).sort_values("_t").reset_index(drop=True)
	info.update(loaded=True, file=fname, rows=len(df))
	return df, info


def _bin_index(grid_ns: np.ndarray, edges_ns: np.ndarray, reading_ns: np.ndarray) -> np.ndarray:
	"""Map each reading time to its HRV bin index, or -1 if outside the timeline."""
	idx = np.searchsorted(edges_ns, reading_ns, side="right") - 1
	idx[(idx < 0) | (idx >= len(grid_ns))] = -1
	return idx


def _aggregate_into_bins(
	grid_ns: np.ndarray,
	edges_ns: np.ndarray,
	reading_ns: np.ndarray,
	values: np.ndarray,
	how: str,
) -> tuple[np.ndarray, np.ndarray]:
	"""Aggregate point readings into HRV bins. Returns (values[N], mask[N])."""
	n = len(grid_ns)
	out = np.full(n, np.nan, dtype=np.float64)
	idx = _bin_index(grid_ns, edges_ns, reading_ns)
	valid = idx >= 0
	if not np.any(valid):
		return out, np.zeros(n, dtype=int)

	binned = pd.DataFrame({"bin": idx[valid], "value": values[valid]})
	grouped = binned.groupby("bin")["value"].mean() if how == "mean" else binned.groupby("bin")["value"].sum()
	out[grouped.index.to_numpy()] = grouped.to_numpy()
	mask = (~np.isnan(out)).astype(int)
	return out, mask


def _asof_within(
	grid: pd.Series, signal_df: pd.DataFrame, value_col: str, tol_min: float
) -> tuple[np.ndarray, np.ndarray]:
	"""Nearest reading within tolerance via merge_asof. Returns (values[N], mask[N])."""
	left = pd.DataFrame({"_t": grid})
	right = signal_df[["_t", value_col]].dropna(subset=["_t"]).sort_values("_t")
	right[value_col] = pd.to_numeric(right[value_col], errors="coerce")
	merged = pd.merge_asof(
		left,
		right,
		on="_t",
		direction="nearest",
		tolerance=pd.Timedelta(minutes=tol_min),
	)
	values = merged[value_col].to_numpy(dtype=np.float64)
	mask = (~np.isnan(values)).astype(int)
	return values, mask


def _is_asleep(grid_ns: np.ndarray, sleep_df: pd.DataFrame) -> np.ndarray:
	"""1 where the HRV timestamp falls inside any [start, end] sleep interval."""
	asleep = np.zeros(len(grid_ns), dtype=int)
	if sleep_df is None or sleep_df.empty:
		return asleep
	starts = sleep_df["_t"].to_numpy(dtype="datetime64[ns]").astype("int64")
	ends = sleep_df["_t_end"].to_numpy(dtype="datetime64[ns]").astype("int64")
	for start, end in zip(starts, ends):
		if end < start:
			start, end = end, start
		asleep |= (grid_ns >= start) & (grid_ns <= end)
	return asleep


def build_fused_timeline(
	signals: dict[str, pd.DataFrame | None],
	temp_tol_min: float,
	spo2_tol_min: float,
) -> tuple[pd.DataFrame, dict]:
	"""Fuse all signals onto the HRV grid. Returns (fused_df, report)."""
	hrv = signals.get("hrv")
	if hrv is None or hrv.empty:
		raise ValueError("HRV signal is required and was not found or is empty.")

	hrv_value_col = _resolve_column(hrv.columns, SIGNALS["hrv"].value_cols, "HRV value")
	hrv = hrv.drop_duplicates(subset=["_t"]).sort_values("_t").reset_index(drop=True)

	grid = hrv["_t"]
	grid_ns = grid.to_numpy(dtype="datetime64[ns]").astype("int64")
	n = len(grid_ns)

	# Synthetic right edge for the last bin: one median HRV gap past the last reading.
	if n > 1:
		median_gap = int(np.median(np.diff(grid_ns)))
	else:
		median_gap = int(10 * 60 * 1e9)  # 10 min fallback
	edges_ns = np.concatenate([grid_ns, [grid_ns[-1] + median_gap]])

	first_ts = grid.iloc[0]
	minute = ((grid - first_ts).dt.total_seconds() / 60.0).round().astype("int64")

	fused = pd.DataFrame(
		{
			"timestamp": grid.dt.strftime("%Y-%m-%d %H:%M:%S"),
			"minute": minute.to_numpy(),
			"hrvValue": pd.to_numeric(hrv[hrv_value_col], errors="coerce").to_numpy(),
		}
	)

	report: dict = {"n_hrv_rows": n, "signals": {}, "coverage": {}}

	# heartrate -> mean per bin
	hr = signals.get("heartrate")
	if hr is not None and not hr.empty:
		hr_val = pd.to_numeric(hr["lastRate"], errors="coerce").to_numpy(dtype=np.float64)
		hr_ns = hr["_t"].to_numpy(dtype="datetime64[ns]").astype("int64")
		fused["HR"], fused["hr_mask"] = _aggregate_into_bins(grid_ns, edges_ns, hr_ns, hr_val, "mean")
	else:
		fused["HR"], fused["hr_mask"] = np.nan, 0

	# steps -> sum per bin (assigned by start time)
	steps = signals.get("steps")
	if steps is not None and not steps.empty:
		st_val = pd.to_numeric(steps["steps"], errors="coerce").to_numpy(dtype=np.float64)
		st_ns = steps["_t"].to_numpy(dtype="datetime64[ns]").astype("int64")
		fused["steps"], fused["steps_mask"] = _aggregate_into_bins(grid_ns, edges_ns, st_ns, st_val, "sum")
	else:
		fused["steps"], fused["steps_mask"] = np.nan, 0

	# temperature -> nearest within tolerance; watch_off flag on raw value
	temp = signals.get("temperature")
	if temp is not None and not temp.empty:
		t_val, t_mask = _asof_within(grid, temp, "temperature", temp_tol_min)
		fused["temperature"], fused["temp_mask"] = t_val, t_mask
		fused["watch_off_flag"] = ((t_val < OFF_SKIN_TEMP) & ~np.isnan(t_val)).astype(int)
	else:
		fused["temperature"], fused["temp_mask"], fused["watch_off_flag"] = np.nan, 0, 0

	# spo2 -> nearest within tolerance; no forward fill
	spo2 = signals.get("spo2")
	if spo2 is not None and not spo2.empty:
		spo2_col = _resolve_column(spo2.columns, SIGNALS["spo2"].value_cols, "spo2 value")
		s_val, s_mask = _asof_within(grid, spo2, spo2_col, spo2_tol_min)
		fused["spo2"], fused["spo2_mask"] = s_val, s_mask
	else:
		fused["spo2"], fused["spo2_mask"] = np.nan, 0

	# sleep -> binary is_asleep (intervals only; description ignored)
	fused["is_asleep"] = _is_asleep(grid_ns, signals.get("sleep"))

	# clip_flag: raw HRV outside a plausible physiological range (flag only;
	# hrvValue / real_value is never altered).
	hv = fused["hrvValue"].to_numpy(dtype=np.float64)
	fused["clip_flag"] = (
		((hv < config.HRV_CLIP_MIN) | (hv > config.HRV_CLIP_MAX)) & ~np.isnan(hv)
	).astype(int)

	# Coverage report (share of HRV rows with a real reading per signal)
	for col, mask_col in [
		("HR", "hr_mask"),
		("steps", "steps_mask"),
		("temperature", "temp_mask"),
		("spo2", "spo2_mask"),
	]:
		report["coverage"][col] = float(np.mean(fused[mask_col].to_numpy())) if n else 0.0
	report["coverage"]["is_asleep"] = float(np.mean(fused["is_asleep"].to_numpy())) if n else 0.0
	report["watch_off_share"] = float(np.mean(fused["watch_off_flag"].to_numpy())) if n else 0.0

	return fused, report


def _format_report(patient: str, infos: dict, report: dict, fused: pd.DataFrame) -> str:
	lines = [f"=== Data-quality report: {patient} ==="]
	ts = pd.to_datetime(fused["timestamp"])
	lines.append(
		f"HRV timeline: {report['n_hrv_rows']} rows, "
		f"{ts.min()} -> {ts.max()} "
		f"({(ts.max() - ts.min()).days} days)"
	)
	lines.append("")
	lines.append(f"{'signal':<12}{'loaded':<8}{'folder':<14}{'raw_rows':<10}{'coverage':<16}{'file'}")
	for name in ["hrv", "heartrate", "temperature", "steps", "sleep", "spo2"]:
		info = infos.get(name, {})
		cov_key = {"heartrate": "HR", "temperature": "temperature"}.get(name, name)
		cov = report["coverage"].get(cov_key)
		if name == "hrv":
			cov_str = "base"
		elif name == "sleep":
			cov_str = f"{report['coverage'].get('is_asleep', 0.0) * 100:.1f}% asleep"
		elif cov is None:
			cov_str = "-"
		else:
			cov_str = f"{cov * 100:.1f}%"
		lines.append(
			f"{name:<12}{str(info.get('loaded', False)):<8}{str(info.get('folder') or '-'):<14}"
			f"{info.get('rows', 0):<10}{cov_str:<16}{info.get('file') or '-'}"
		)
	lines.append("")
	lines.append(f"watch_off (temp<{OFF_SKIN_TEMP:.0f}F): {report['watch_off_share'] * 100:.1f}% of rows")
	return "\n".join(lines)


def ingest_patient(
	patient_dir: Path,
	output_dir: Path,
	temp_tol_min: float = DEFAULT_TEMP_TOL_MIN,
	spo2_tol_min: float = DEFAULT_SPO2_TOL_MIN,
	days: float | None = None,
) -> tuple[Path, pd.DataFrame, str]:
	"""Ingest one patient folder. Returns (output_csv, fused_df, report_text)."""
	signals: dict[str, pd.DataFrame | None] = {}
	infos: dict[str, dict] = {}
	for name, spec in SIGNALS.items():
		df, info = load_signal(patient_dir, spec)
		signals[name] = df
		infos[name] = info
		state = f"{info['rows']} rows from {info['file']}" if info["loaded"] else "MISSING"
		print(f"  [{name:<12}] {state}")

	# Flag any folders we did not recognise (e.g. realtimetemphr).
	known = {f for spec in SIGNALS.values() for f in spec.folders}
	for sub in sorted(p.name for p in patient_dir.iterdir() if p.is_dir()):
		if sub not in known:
			print(f"  [skip] unrecognised signal folder: {sub}")

	fused, report = build_fused_timeline(signals, temp_tol_min, spo2_tol_min)
	report_text = _format_report(patient_dir.name, infos, report, fused)

	safe_id = re.sub(r'[^A-Za-z0-9_-]', "", patient_dir.name) or "patient"
	output_dir.mkdir(parents=True, exist_ok=True)
	out_csv = output_dir / f"{safe_id}_aligned.csv"
	fused.to_csv(out_csv, index=False, na_rep="")
	(output_dir / f"{safe_id}_quality.txt").write_text(report_text + "\n")

	if days is not None:
		# Previews go in a subdir so the pipeline's `*.csv` glob over the fused dir
		# only ever picks up real per-patient inputs.
		preview_dir = output_dir / "previews"
		preview_dir.mkdir(parents=True, exist_ok=True)
		ts = pd.to_datetime(fused["timestamp"])
		cutoff = ts.iloc[0] + pd.Timedelta(days=days)
		preview = fused[ts <= cutoff]
		preview_csv = preview_dir / f"{safe_id}_preview_{int(days)}d.csv"
		preview.to_csv(preview_csv, index=False, na_rep="")
		print(f"\nPreview ({len(preview)} rows, first {days} days) -> {preview_csv.relative_to(output_dir)}")
		cols = ["timestamp", "minute", "hrvValue", "HR", "steps", "temperature", "watch_off_flag", "spo2", "is_asleep"]
		with pd.option_context("display.max_rows", 30, "display.width", 200):
			print(preview[cols].head(30).to_string(index=False))

	return out_csv, fused, report_text


def _resolve_patient_dir(exports_dir: Path, patient: str) -> Path:
	direct = exports_dir / patient
	if direct.is_dir():
		return direct
	# Patient folders may carry literal quotes in their names; match leniently.
	target = re.sub(r'[^A-Za-z0-9]', "", patient).lower()
	for p in exports_dir.iterdir():
		if p.is_dir() and re.sub(r'[^A-Za-z0-9]', "", p.name).lower() == target:
			return p
	raise FileNotFoundError(f"Patient folder not found for '{patient}' in {exports_dir}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Fuse wearable signals onto the HRV timeline.")
	parser.add_argument("--exports-dir", default=str(EXPORTS_DIR), help="Dir of patient folders")
	parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Dir for fused CSVs")
	parser.add_argument("--patient", default=None, help="Patient folder name (e.g. a001)")
	parser.add_argument("--all", action="store_true", help="Process every patient folder")
	parser.add_argument("--temp-tol", type=float, default=DEFAULT_TEMP_TOL_MIN, help="Temperature asof tolerance (min)")
	parser.add_argument("--spo2-tol", type=float, default=DEFAULT_SPO2_TOL_MIN, help="SpO2 asof tolerance (min)")
	parser.add_argument("--days", type=float, default=None, help="Also emit a preview of the first N days")
	args = parser.parse_args()

	exports_dir = Path(args.exports_dir)
	output_dir = Path(args.output_dir)
	if not exports_dir.is_dir():
		raise FileNotFoundError(f"Exports directory not found: {exports_dir}")

	if args.all:
		patient_dirs = sorted((p for p in exports_dir.iterdir() if p.is_dir()), key=lambda p: p.name)
	elif args.patient:
		patient_dirs = [_resolve_patient_dir(exports_dir, args.patient)]
	else:
		parser.error("Specify --patient <id> or --all")

	for patient_dir in patient_dirs:
		print(f"\n=== Ingesting {patient_dir.name} ===")
		try:
			out_csv, _, report_text = ingest_patient(
				patient_dir, output_dir, args.temp_tol, args.spo2_tol, args.days
			)
			print("\n" + report_text)
			print(f"\nSaved: {out_csv}")
		except Exception as exc:
			print(f"  Failed: {patient_dir.name} ({exc})")


if __name__ == "__main__":
	main()
