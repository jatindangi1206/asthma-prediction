import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import multiprocessing
import warnings
warnings.filterwarnings("ignore")


def plot_patient(processed_csv, output_dir, gap_minutes=180):
    """4-panel diagnostic plot: raw, raw+gaps, smoothed+CI, trend."""
    try:
        df = pd.read_csv(processed_csv)
        df['createdTime'] = pd.to_datetime(df['createdTime'])
        df = df.sort_values('createdTime').reset_index(drop=True)

        df['time_diff'] = df['createdTime'].diff()
        df['is_new_segment'] = (df['time_diff'] > pd.Timedelta(minutes=gap_minutes)).astype(int)
        gap_times = df.loc[df['is_new_segment'] == 1, 'createdTime']

        t = df['createdTime']
        y = df['hrvValue']

        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

        # Panel 1: Raw
        ax = axes[0]
        ax.scatter(t, y, s=3, color='red', alpha=0.5)
        ax.set_title(f'{processed_csv.stem}  —  Raw HRV  (n={len(df)})', fontsize=11)
        ax.set_ylabel('HRV')
        ax.grid(alpha=0.3)

        # Panel 2: Raw + Gaps
        ax = axes[1]
        ax.scatter(t, y, s=3, color='red', alpha=0.5, label='Observed')
        for gt in gap_times:
            ax.axvline(gt, color='orange', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.set_title(f'Raw + Gap Markers (>{gap_minutes} min, n_gaps={len(gap_times)})', fontsize=11)
        ax.set_ylabel('HRV')
        ax.grid(alpha=0.3)
        if len(gap_times) > 0:
            ax.plot([], [], color='orange', linestyle='--', label=f'Gap >{gap_minutes}min')
        ax.legend(loc='upper right', fontsize=9)

        # Panel 3: Smoothed + 95% CI
        ax = axes[2]
        ax.fill_between(t, df['ci_lower_95'], df['ci_upper_95'],
                        alpha=0.25, color='#22409A', label='95% FFBS CI')
        ax.scatter(t, y, s=2, color='red', alpha=0.3, label='Observed')
        ax.plot(t, df['smoothed_hrv'], color='#22409A', linewidth=1.0,
                label='Smoothed (level + seasonal)')
        for gt in gap_times:
            ax.axvline(gt, color='orange', linestyle='--', linewidth=0.6, alpha=0.4)
        ax.set_title('Smoothed signal (level + circadian) with 95% credible band', fontsize=11)
        ax.set_ylabel('HRV')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)

        # Panel 4: Trend only
        ax = axes[3]
        ax.scatter(t, y, s=2, color='red', alpha=0.2, label='Observed (faded)')
        ax.plot(t, df['true_trend_level'], color='#0B7B3E', linewidth=1.5,
                label='Trend (level only, circadian removed)')
        for gt in gap_times:
            ax.axvline(gt, color='orange', linestyle='--', linewidth=0.6, alpha=0.4)
        ax.set_title('Underlying trend (baseline drift, no circadian)', fontsize=11)
        ax.set_ylabel('HRV')
        ax.set_xlabel('Time')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)

        axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
        axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator()))

        plt.tight_layout()
        out_file = Path(output_dir) / f"{processed_csv.stem}.png"
        plt.savefig(out_file, dpi=110, bbox_inches='tight')
        plt.close(fig)

        return {'file': processed_csv.name, 'status': 'success'}

    except Exception as e:
        plt.close('all')
        return {'file': processed_csv.name, 'status': 'failed', 'reason': str(e)}


def _worker(args):
    return plot_patient(*args)


def run_visualization(base_directory):
    base_path = Path(base_directory)
    processed_dir = base_path / "processed"
    plots_dir = base_path / "plots"
    results_dir = base_path / "results"

    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(processed_dir.glob("*_processed.csv"))
    if not csv_files:
        print(f"ERROR: No processed CSVs found in {processed_dir}")
        return

    print(f"\n--- PHASE 3: VISUALIZATION ---")
    print(f"Source: {processed_dir.resolve()}")
    print(f"Target: {len(csv_files)} processed patient files")
    print(f"Output: {plots_dir.resolve()}")

    max_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {max_workers} CPU cores...\n")

    args_list = [(f, plots_dir) for f in csv_files]
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, a): a for a in args_list}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(args_list), desc="Generating plots"):
            results.append(future.result())

    df_log = pd.DataFrame(results)
    log_path = results_dir / "phase3_plot_log.csv"
    df_log.to_csv(log_path, index=False)

    successes = df_log[df_log['status'] == 'success']
    failures = df_log[df_log['status'] == 'failed']

    print("\n" + "=" * 60)
    print("PHASE 3 COMPLETE")
    print("=" * 60)
    print(f"Plots: {len(successes)} / {len(csv_files)}  |  Failed: {len(failures)}")
    print(f"PNGs:  {plots_dir.resolve()}")
    print(f"Log:   {log_path.resolve()}")

    if len(failures) > 0:
        print("\nFirst 5 failures:")
        for _, row in failures.head(5).iterrows():
            print(f"  - {row['file']}: {row['reason']}")


if __name__ == "__main__":
    BASE_DATA_DIR = "./data"
    run_visualization(BASE_DATA_DIR)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import multiprocessing
import warnings
warnings.filterwarnings("ignore")


def plot_patient(processed_csv, output_dir, gap_minutes=180):
    """4-panel diagnostic plot: raw, raw+gaps, smoothed+CI, trend."""
    try:
        df = pd.read_csv(processed_csv)
        df['createdTime'] = pd.to_datetime(df['createdTime'])
        df = df.sort_values('createdTime').reset_index(drop=True)

        df['time_diff'] = df['createdTime'].diff()
        df['is_new_segment'] = (df['time_diff'] > pd.Timedelta(minutes=gap_minutes)).astype(int)
        gap_times = df.loc[df['is_new_segment'] == 1, 'createdTime']

        t = df['createdTime']
        y = df['hrvValue']

        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

        # Panel 1: Raw
        ax = axes[0]
        ax.scatter(t, y, s=3, color='red', alpha=0.5)
        ax.set_title(f'{processed_csv.stem}  —  Raw HRV  (n={len(df)})', fontsize=11)
        ax.set_ylabel('HRV'); ax.grid(alpha=0.3)

        # Panel 2: Raw + Gaps
        ax = axes[1]
        ax.scatter(t, y, s=3, color='red', alpha=0.5, label='Observed')
        for gt in gap_times:
            ax.axvline(gt, color='orange', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.set_title(f'Raw + Gap Markers (>{gap_minutes} min, n_gaps={len(gap_times)})', fontsize=11)
        ax.set_ylabel('HRV'); ax.grid(alpha=0.3)
        if len(gap_times) > 0:
            ax.plot([], [], color='orange', linestyle='--', label=f'Gap >{gap_minutes}min')
        ax.legend(loc='upper right', fontsize=9)

        # Panel 3: Smoothed + 95% CI
        ax = axes[2]
        ax.fill_between(t, df['ci_lower_95'], df['ci_upper_95'],
                        alpha=0.25, color='#22409A', label='95% FFBS CI')
        ax.scatter(t, y, s=2, color='red', alpha=0.3, label='Observed')
        ax.plot(t, df['smoothed_hrv'], color='#22409A', linewidth=1.0,
                label='Smoothed (level + seasonal)')
        for gt in gap_times:
            ax.axvline(gt, color='orange', linestyle='--', linewidth=0.6, alpha=0.4)
        ax.set_title('Smoothed signal (level + circadian) with 95% credible band', fontsize=11)
        ax.set_ylabel('HRV'); ax.legend(loc='upper right', fontsize=9); ax.grid(alpha=0.3)

        # Panel 4: Trend only
        ax = axes[3]
        ax.scatter(t, y, s=2, color='red', alpha=0.2, label='Observed (faded)')
        ax.plot(t, df['true_trend_level'], color='#0B7B3E', linewidth=1.5,
                label='Trend (level only, circadian removed)')
        for gt in gap_times:
            ax.axvline(gt, color='orange', linestyle='--', linewidth=0.6, alpha=0.4)
        ax.set_title('Underlying trend (baseline drift, no circadian)', fontsize=11)
        ax.set_ylabel('HRV'); ax.set_xlabel('Time')
        ax.legend(loc='upper right', fontsize=9); ax.grid(alpha=0.3)

        axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
        axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator()))

        plt.tight_layout()
        out_file = Path(output_dir) / f"{processed_csv.stem}.png"
        plt.savefig(out_file, dpi=110, bbox_inches='tight')
        plt.close(fig)

        return {'file': processed_csv.name, 'status': 'success'}

    except Exception as e:
        plt.close('all')
        return {'file': processed_csv.name, 'status': 'failed', 'reason': str(e)}


def _worker(args):
    return plot_patient(*args)


def run_visualization(base_directory):
    base_path = Path(base_directory)
    processed_dir = base_path / "processed"
    plots_dir = base_path / "plots"
    results_dir = base_path / "results"

    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(processed_dir.glob("*_processed.csv"))
    if not csv_files:
        print(f"ERROR: No processed CSVs found in {processed_dir}")
        return

    print(f"\n--- PHASE 3: VISUALIZATION ---")
    print(f"Source: {processed_dir.resolve()}")
    print(f"Target: {len(csv_files)} processed patient files")
    print(f"Output: {plots_dir.resolve()}")

    max_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {max_workers} CPU cores...\n")

    args_list = [(f, plots_dir) for f in csv_files]
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, a): a for a in args_list}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(args_list), desc="Generating plots"):
            results.append(future.result())

    df_log = pd.DataFrame(results)
    log_path = results_dir / "phase3_plot_log.csv"
    df_log.to_csv(log_path, index=False)

    successes = df_log[df_log['status'] == 'success']
    failures = df_log[df_log['status'] == 'failed']

    print("\n" + "=" * 60)
    print("PHASE 3 COMPLETE")
    print("=" * 60)
    print(f"Plots: {len(successes)} / {len(csv_files)}  |  Failed: {len(failures)}")
    print(f"PNGs:  {plots_dir.resolve()}")
    print(f"Log:   {log_path.resolve()}")

    if len(failures) > 0:
        print("\nFirst 5 failures:")
        for _, row in failures.head(5).iterrows():
            print(f"  - {row['file']}: {row['reason']}")


if __name__ == "__main__":
    BASE_DATA_DIR = "./data"
    run_visualization(BASE_DATA_DIR)