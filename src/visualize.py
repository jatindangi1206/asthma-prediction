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


# ─────────────────────────────────────────────────────────────────────────────
# CPD columns confirmed from the schema report.
# Each method writes  <method>_type   in {'normal','transition','shift'}
# and                  <method>_degree in [0, 1].
# Change points fire on  <method>_type == 'shift'.
# ─────────────────────────────────────────────────────────────────────────────
CPD_METHODS = {
    'BOCPD':  {'type_col': 'bocpd_type',  'degree_col': 'bocpd_degree'},
    'Kalman': {'type_col': 'kalman_type', 'degree_col': 'kalman_degree'},
    'KCP':    {'type_col': 'kcp_type',    'degree_col': 'kcp_degree'},
    'HMM':    {'type_col': 'hmm_type',    'degree_col': 'hmm_degree'},
}

METHOD_COLORS = {
    'BOCPD':  '#D7263D',
    'Kalman': '#1B998B',
    'KCP':    '#F46036',
    'HMM':    '#5D2E8C',
}


def _load_patient(annotated_csv, gap_minutes):
    df = pd.read_csv(annotated_csv)
    df['createdTime'] = pd.to_datetime(df['createdTime'])
    df = df.sort_values('createdTime').reset_index(drop=True)
    df['time_diff'] = df['createdTime'].diff()
    df['is_new_segment'] = (df['time_diff'] > pd.Timedelta(minutes=gap_minutes)).astype(int)
    return df


def _shift_times(df, type_col):
    if type_col not in df.columns:
        return pd.Series([], dtype='datetime64[ns]')
    return df.loc[df[type_col] == 'shift', 'createdTime']


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Original 4-panel diagnostic, with CPD shift markers on the trend panel
# ─────────────────────────────────────────────────────────────────────────────
def _diagnostic_4panel(df, stem, output_dir, gap_minutes):
    t = df['createdTime']
    y = df['hrvValue']
    gap_times = df.loc[df['is_new_segment'] == 1, 'createdTime']

    has_ci       = ('ci_lower_95' in df.columns) and ('ci_upper_95' in df.columns)
    has_smoothed = 'smoothed_hrv' in df.columns
    has_trend    = 'true_trend_level' in df.columns

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    # Panel 1: Raw
    ax = axes[0]
    ax.scatter(t, y, s=3, color='red', alpha=0.5)
    ax.set_title(f'{stem}  —  Raw HRV  (n={len(df)})', fontsize=11)
    ax.set_ylabel('HRV')
    ax.grid(alpha=0.3)

    # Panel 2: Raw + gap markers
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
    if has_ci:
        ax.fill_between(t, df['ci_lower_95'], df['ci_upper_95'],
                        alpha=0.25, color='#22409A', label='95% FFBS CI')
    ax.scatter(t, y, s=2, color='red', alpha=0.3, label='Observed')
    if has_smoothed:
        ax.plot(t, df['smoothed_hrv'], color='#22409A', linewidth=1.0,
                label='Smoothed (level + seasonal)')
    for gt in gap_times:
        ax.axvline(gt, color='orange', linestyle='--', linewidth=0.6, alpha=0.4)
    ax.set_title('Smoothed signal (level + circadian) with 95% credible band', fontsize=11)
    ax.set_ylabel('HRV')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 4: Trend + CPD shift markers from all four methods
    ax = axes[3]
    ax.scatter(t, y, s=2, color='red', alpha=0.2, label='Observed (faded)')
    if has_trend:
        ax.plot(t, df['true_trend_level'], color='#0B7B3E', linewidth=1.5,
                label='Trend (level only, circadian removed)')
    elif has_smoothed:
        ax.plot(t, df['smoothed_hrv'], color='#0B7B3E', linewidth=1.5,
                label='Smoothed (trend fallback)')
    for gt in gap_times:
        ax.axvline(gt, color='orange', linestyle='--', linewidth=0.6, alpha=0.3)

    # CPD shift markers, one color per method, on the trend
    for method, spec in CPD_METHODS.items():
        cps = _shift_times(df, spec['type_col'])
        for cp in cps:
            ax.axvline(cp, color=METHOD_COLORS[method], linewidth=1.0, alpha=0.75)
        if len(cps) > 0:
            ax.plot([], [], color=METHOD_COLORS[method], linewidth=1.2,
                    label=f'{method} shift (n={len(cps)})')

    ax.set_title('Underlying trend (baseline drift, no circadian) — with CPD shift annotations',
                 fontsize=11)
    ax.set_ylabel('HRV')
    ax.set_xlabel('Time')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1].xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator())
    )

    plt.tight_layout()
    out_file = Path(output_dir) / f"{stem}_diagnostic.png"
    plt.savefig(out_file, dpi=110, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure set 2: Per-method plots (one figure per CPD method)
# ─────────────────────────────────────────────────────────────────────────────
def _per_method_plot(df, method_name, type_col, color, stem, output_dir, gap_minutes):
    t = df['createdTime']
    y = df['hrvValue']
    gap_times = df.loc[df['is_new_segment'] == 1, 'createdTime']
    shift_times = _shift_times(df, type_col)

    fig, ax = plt.subplots(figsize=(16, 4.5))
    ax.scatter(t, y, s=2, color='red', alpha=0.25, label='Observed')
    if 'smoothed_hrv' in df.columns:
        ax.plot(t, df['smoothed_hrv'], color='#22409A', linewidth=1.0, label='Smoothed')

    for gt in gap_times:
        ax.axvline(gt, color='orange', linestyle='--', linewidth=0.5, alpha=0.4)
    for cp in shift_times:
        ax.axvline(cp, color=color, linewidth=1.2, alpha=0.85)
    if len(shift_times) > 0:
        ax.plot([], [], color=color, linewidth=1.2,
                label=f'{method_name} shift (n={len(shift_times)})')

    ax.set_title(f'{stem}  —  {method_name} change points on smoothed HRV', fontsize=11)
    ax.set_ylabel('HRV')
    ax.set_xlabel('Time')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    plt.tight_layout()
    out_file = Path(output_dir) / f"{stem}_{method_name.lower()}.png"
    plt.savefig(out_file, dpi=110, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Consolidated overlay (main panel + per-method rug tracks + union)
# ─────────────────────────────────────────────────────────────────────────────
def _overlay_plot(df, stem, output_dir, gap_minutes):
    t = df['createdTime']
    y = df['hrvValue']
    gap_times = df.loc[df['is_new_segment'] == 1, 'createdTime']

    method_cps = {
        name: _shift_times(df, spec['type_col'])
        for name, spec in CPD_METHODS.items()
    }
    union_cps = pd.concat(list(method_cps.values())).drop_duplicates().sort_values()

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(6, 1, height_ratios=[6, 0.5, 0.5, 0.5, 0.5, 0.6], hspace=0.15)

    ax_main = fig.add_subplot(gs[0, 0])
    ax_main.scatter(t, y, s=2, color='red', alpha=0.20, label='Raw HRV')
    for gt in gap_times:
        ax_main.axvline(gt, color='orange', linestyle='--', linewidth=0.6, alpha=0.5)
    if len(gap_times) > 0:
        ax_main.plot([], [], color='orange', linestyle='--', label=f'Gap >{gap_minutes}min')
    if 'true_trend_level' in df.columns:
        ax_main.plot(t, df['true_trend_level'], color='#0B7B3E', linewidth=1.6,
                     label='Smoothed trend (level only)')
    elif 'smoothed_hrv' in df.columns:
        ax_main.plot(t, df['smoothed_hrv'], color='#0B7B3E', linewidth=1.4,
                     label='Smoothed HRV')
    ax_main.set_title(f'{stem}  —  Layered overlay: raw → gaps → smoothed → CPD', fontsize=12)
    ax_main.set_ylabel('HRV')
    ax_main.legend(loc='upper right', fontsize=9)
    ax_main.grid(alpha=0.3)

    rug_specs = [
        ('BOCPD',  METHOD_COLORS['BOCPD'],  method_cps['BOCPD']),
        ('Kalman', METHOD_COLORS['Kalman'], method_cps['Kalman']),
        ('KCP',    METHOD_COLORS['KCP'],    method_cps['KCP']),
        ('HMM',    METHOD_COLORS['HMM'],    method_cps['HMM']),
        ('Union',  '#222222',               union_cps),
    ]

    rug_axes = []
    for i, (name, color, cps) in enumerate(rug_specs, start=1):
        ax = fig.add_subplot(gs[i, 0], sharex=ax_main)
        for cp in cps:
            ax.axvline(cp, color=color, linewidth=1.0, alpha=0.9)
        ax.set_yticks([])
        ax.set_ylabel(f'{name}\n(n={len(cps)})', rotation=0, ha='right', va='center', fontsize=9)
        ax.grid(alpha=0.2, axis='x')
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        if i < len(rug_specs):
            ax.tick_params(labelbottom=False)
        rug_axes.append(ax)

    rug_axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    rug_axes[-1].xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(rug_axes[-1].xaxis.get_major_locator())
    )
    rug_axes[-1].set_xlabel('Time')

    out_file = Path(output_dir) / f"{stem}_overlay.png"
    plt.savefig(out_file, dpi=110, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Per-patient driver
# ─────────────────────────────────────────────────────────────────────────────
def plot_patient(annotated_csv, output_dir, gap_minutes=180):
    """Produces per patient:
       - 1 diagnostic 4-panel (raw / raw+gaps / smoothed+CI / trend+CPD)
       - 4 per-method change-point plots
       - 1 consolidated overlay
    """
    try:
        df = _load_patient(annotated_csv, gap_minutes)
        stem = annotated_csv.stem.replace('_annotated', '')

        _diagnostic_4panel(df, stem, output_dir, gap_minutes)

        for method, spec in CPD_METHODS.items():
            _per_method_plot(df, method, spec['type_col'], METHOD_COLORS[method],
                             stem, output_dir, gap_minutes)

        _overlay_plot(df, stem, output_dir, gap_minutes)

        return {'file': annotated_csv.name, 'status': 'success'}

    except Exception as e:
        plt.close('all')
        return {'file': annotated_csv.name, 'status': 'failed', 'reason': str(e)}


def _worker(args):
    return plot_patient(*args)


def run_visualization(base_directory):
    base_path = Path(base_directory)
    annotated_dir = base_path / "annotated"
    plots_dir     = base_path / "plots"
    results_dir   = base_path / "results"

    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(annotated_dir.glob("*_annotated.csv"))
    if not csv_files:
        print(f"ERROR: No annotated CSVs found in {annotated_dir}")
        print(f"       Run src/03_annotate.py first to produce ./data/annotated/*_annotated.csv")
        return

    print(f"\n--- PHASE 3: VISUALIZATION ---")
    print(f"Source: {annotated_dir.resolve()}")
    print(f"Target: {len(csv_files)} annotated patient files (6 figures each)")
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
    failures  = df_log[df_log['status'] == 'failed']

    print("\n" + "=" * 60)
    print("PHASE 3 COMPLETE")
    print("=" * 60)
    print(f"Patients done: {len(successes)} / {len(csv_files)}  |  Failed: {len(failures)}")
    print(f"PNGs:  {plots_dir.resolve()}")
    print(f"Log:   {log_path.resolve()}")

    if len(failures) > 0:
        print("\nFirst 5 failures:")
        for _, row in failures.head(5).iterrows():
            print(f"  - {row['file']}: {row['reason']}")


if __name__ == "__main__":
    BASE_DATA_DIR = "./data"
    run_visualization(BASE_DATA_DIR)

