import os
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import multiprocessing

warnings.filterwarnings("ignore")

import particles
from particles import distributions as dists
from particles import state_space_models as ssm

INPUT_DIR = Path("./data/processed")
OUTPUT_DIR = Path("./data/smoothed")
RESULTS_DIR = Path("./data/results")

# Injected from Phase 1 (01_hyper_tuning.py) results
GLOBAL_N = 250
GLOBAL_M = 50

# ==============================================================================
# 1. STATE-SPACE MODEL
# ==============================================================================
class HRVParticleModel(ssm.StateSpaceModel):
    default_params = {
        'sigma_level': 0.5, 'sigma_slope': 0.02, 'sigma_seas': 0.15,
        'sigma_obs': 12.0, 'nu': 4.0,
        'init_level_loc': 95.0, 'init_level_scale': 30.0,
        'omega1': 2 * np.pi / 144, 'omega2': 2 * np.pi / 72,
        'dt_norm': None,
    }

    def PX0(self):
        return dists.StructDist({
            'level':   dists.Normal(loc=self.init_level_loc, scale=self.init_level_scale),
            'slope':   dists.Normal(loc=0.0, scale=0.3),
            'c1':      dists.Normal(loc=0.0, scale=35.0),
            'c1_star': dists.Normal(loc=0.0, scale=35.0),
            'c2':      dists.Normal(loc=0.0, scale=15.0),
            'c2_star': dists.Normal(loc=0.0, scale=15.0),
        })

    def PX(self, t, xp):
        dt = self.dt_norm[t] if self.dt_norm is not None else 1.0
        dt_sqrt = np.sqrt(dt)

        level_mean = xp['level'] + xp['slope'] * dt
        a1, b1 = np.cos(self.omega1 * dt), np.sin(self.omega1 * dt)
        a2, b2 = np.cos(self.omega2 * dt), np.sin(self.omega2 * dt)

        return dists.StructDist({
            'level':   dists.Normal(loc=level_mean,                                scale=self.sigma_level * dt_sqrt),
            'slope':   dists.Normal(loc=xp['slope'],                               scale=self.sigma_slope * dt_sqrt),
            'c1':      dists.Normal(loc=xp['c1'] * a1 + xp['c1_star'] * b1,       scale=self.sigma_seas),
            'c1_star': dists.Normal(loc=-xp['c1'] * b1 + xp['c1_star'] * a1,      scale=self.sigma_seas),
            'c2':      dists.Normal(loc=xp['c2'] * a2 + xp['c2_star'] * b2,       scale=self.sigma_seas),
            'c2_star': dists.Normal(loc=-xp['c2'] * b2 + xp['c2_star'] * a2,      scale=self.sigma_seas),
        })

    def PY(self, t, xp, x):
        observable_mean = x['level'] + x['c1'] + x['c2']
        return dists.Student(df=self.nu, loc=observable_mean, scale=self.sigma_obs)


class HRVBootstrap(ssm.Bootstrap):
    """Bootstrap filter that treats NaN observations as missing.

    In particles, the data lives on the FK (here), not on the ssm, so NaN
    handling must happen in logG. logpdf(NaN) is always NaN regardless of
    scale, and a single NaN log-weight corrupts every particle weight and
    propagates downstream. We instead return uniform (zero) log-weights at
    missing observations, so the filter propagates on the dynamics alone.
    """

    def logG(self, t, xp, x):
        y_t = self.data[t]
        if np.isnan(y_t):
            return np.zeros(x['level'].shape[0])
        return self.ssm.PY(t, xp, x).logpdf(y_t)


def compute_data_driven_params(y_data, median_dt_minutes):
    y = y_data.astype(float)
    n_per_day = max(int(24 * 60 / median_dt_minutes), 50)
    first_day_clean = y[:n_per_day][~np.isnan(y[:n_per_day])]

    if len(first_day_clean) > 5:
        init_level_loc = float(np.median(first_day_clean))
        init_level_scale = float(max(2 * np.median(np.abs(first_day_clean - init_level_loc)), 10.0))
    else:
        init_level_loc = float(np.nanmedian(y))
        init_level_scale = 20.0

    diffs = np.diff(y)
    diffs = diffs[~np.isnan(diffs)]
    mad_diffs = np.median(np.abs(diffs - np.median(diffs))) if len(diffs) > 0 else 5.0

    sigma_obs = max(float(1.4826 * mad_diffs / np.sqrt(2)), 3.0)
    sigma_level = 0.05 * sigma_obs
    sigma_slope = 0.02 * sigma_level
    obs_per_day = 24 * 60 / median_dt_minutes

    return {
        'init_level_loc':   init_level_loc,
        'init_level_scale': init_level_scale,
        'sigma_obs':        sigma_obs,
        'sigma_level':      sigma_level,
        'sigma_slope':      sigma_slope,
        'sigma_seas':       0.15,
        'omega1':           float(2 * np.pi / obs_per_day),
        'omega2':           float(2 * np.pi / (obs_per_day / 2)),
    }


# ==============================================================================
# 2. WORKER
# ==============================================================================
def process_patient(args):
    file_path, output_dir, N_particles, M_paths = args
    try:
        if os.path.getsize(file_path) < 10:
            return {'file': file_path.name, 'status': 'failed', 'reason': 'Empty file'}

        df = pd.read_csv(file_path, encoding='utf-8-sig')
        df.columns = df.columns.str.strip()

        required = {'createdTime', 'hrvValue', 'minute_diff'}
        if not required.issubset(df.columns):
            return {'file': file_path.name, 'status': 'failed',
                    'reason': f'Missing columns. Found: {list(df.columns)}'}

        df['createdTime'] = pd.to_datetime(df['createdTime'])
        df = df.sort_values('createdTime').reset_index(drop=True)

        y_data = df['hrvValue'].values.astype(float)
        T = len(y_data)

        if T < 10:
            return {'file': file_path.name, 'status': 'failed', 'reason': 'Too short (<10 rows)'}

        # Use pre-computed minute_diff; treat first row as nominal step
        minute_diff = df['minute_diff'].values.astype(float)
        minute_diff[0] = 0.0
        valid_diffs = minute_diff[minute_diff > 0]
        median_dt = float(np.median(valid_diffs)) if len(valid_diffs) > 0 else 10.0

        dt_norm = minute_diff / median_dt
        dt_norm[0] = 1.0
        dt_norm = np.where(dt_norm <= 0, 1.0, dt_norm)  # guard zero/negative

        params = compute_data_driven_params(y_data, median_dt)
        params['dt_norm'] = dt_norm

        fk = HRVBootstrap(ssm=HRVParticleModel(**params), data=y_data)
        alg = particles.SMC(fk=fk, N=N_particles, resampling='systematic',
                            store_history=True, verbose=False)
        alg.run()

        trajectories = alg.hist.backward_sampling_ON2(M_paths)

        observable_paths = np.zeros((M_paths, T))
        for t in range(T):
            observable_paths[:, t] = (
                trajectories[t]['level'] + trajectories[t]['c1'] + trajectories[t]['c2']
            )

        df['smoothed_hrv'] = np.mean(observable_paths, axis=0)

        out_cols = ['createdTime', 'hrvValue', 'minute_diff', 'smoothed_hrv']
        stem = file_path.stem.replace('_processed', '')
        out_file = Path(output_dir) / f"{stem}_smoothed.csv"
        df[out_cols].to_csv(out_file, index=False)

        return {'file': file_path.name, 'status': 'success', 'n_rows': T}

    except Exception as e:
        return {'file': file_path.name, 'status': 'failed', 'reason': str(e)}


# ==============================================================================
# 3. ORCHESTRATION
# ==============================================================================
def run_smoothing(global_N=GLOBAL_N, global_M=GLOBAL_M):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(INPUT_DIR.glob("*_processed.csv"))
    if not csv_files:
        print(f"ERROR: No processed CSVs found in {INPUT_DIR.resolve()}")
        return

    print(f"\n--- SMC FILTER ---")
    print(f"Input:      {INPUT_DIR.resolve()}")
    print(f"Output:     {OUTPUT_DIR.resolve()}")
    print(f"Patients:   {len(csv_files)}")
    print(f"Particles N={global_N}  Paths M={global_M}")

    max_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Workers:    {max_workers}\n")

    args_list = [(f, OUTPUT_DIR, global_N, global_M) for f in csv_files]
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_patient, a): a for a in args_list}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(args_list), desc="Smoothing"):
            results.append(future.result())

    df_log = pd.DataFrame(results)
    log_path = RESULTS_DIR / "smoothing_log.csv"
    df_log.to_csv(log_path, index=False)

    successes = df_log[df_log['status'] == 'success']
    failures  = df_log[df_log['status'] == 'failed']

    print("\n" + "=" * 60)
    print("SMOOTHING COMPLETE")
    print("=" * 60)
    print(f"Success: {len(successes)} / {len(csv_files)}")
    print(f"Failed:  {len(failures)}")
    print(f"Output:  {OUTPUT_DIR.resolve()}")
    print(f"Log:     {log_path.resolve()}")

    if len(failures) > 0:
        print("\nFailures:")
        for _, row in failures.iterrows():
            print(f"  - {row['file']}: {row['reason']}")


if __name__ == "__main__":
    run_smoothing()
