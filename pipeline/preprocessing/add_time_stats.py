"""Add minute_diff and mean columns to raw HRV CSVs.

For each CSV, if `minute_diff` is missing it is computed as:
    minute_diff[i] = minute[i] - minute[i-1], minute_diff[0] = 0

If `mean` is missing, it is computed as the file-level mean of minute_diff.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "raw_data"


def _resolve_column(columns: pd.Index, candidates: list[str], label: str) -> str:
	for name in candidates:
		if name in columns:
			return name
	raise ValueError(f"Missing {label} column. Expected one of: {', '.join(candidates)}")


def add_time_stats_if_missing(csv_path: Path) -> bool:
	"""Add missing time stats columns in-place. Returns True if updated."""
	df = pd.read_csv(csv_path)
	updated = False

	# Compute minute_diff only if missing.
	minute_diff_col = "minute_diff" if "minute_diff" in df.columns else None
	if minute_diff_col is None:
		time_col = _resolve_column(df.columns, ["minute", "time", "timestamp"], "time")
		minute_values = pd.to_numeric(df[time_col], errors="coerce")
		if minute_values.isna().any():
			raise ValueError(f"Non-numeric time values found in {csv_path.name}")
		df["minute_diff"] = minute_values.diff().fillna(0.0)
		minute_diff_col = "minute_diff"
		updated = True

	# Compute mean only if missing.
	if "mean" not in df.columns:
		mean_value = df[minute_diff_col].mean()
		df["mean"] = mean_value
		updated = True

	if updated:
		df.to_csv(csv_path, index=False)

	return updated


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Add minute_diff and mean columns to raw HRV CSVs."
	)
	parser.add_argument(
		"--input-dir",
		default=str(RAW_DATA_DIR),
		help="Directory containing raw CSVs (default: pipeline/raw_data)",
	)
	args = parser.parse_args()

	input_dir = Path(args.input_dir)
	if not input_dir.exists():
		raise FileNotFoundError(f"Input directory not found: {input_dir}")

	csv_files = sorted(input_dir.glob("*.csv"))
	if not csv_files:
		print(f"No CSV files found in {input_dir}")
		return

	updated_count = 0
	for csv_file in csv_files:
		try:
			if add_time_stats_if_missing(csv_file):
				updated_count += 1
				print(f"Updated: {csv_file.name}")
			else:
				print(f"Skipped (already has columns): {csv_file.name}")
		except Exception as exc:
			print(f"Failed: {csv_file.name} ({exc})")

	print(f"Done. Updated {updated_count}/{len(csv_files)} files.")


if __name__ == "__main__":
	main()
