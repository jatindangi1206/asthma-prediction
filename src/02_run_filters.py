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
from scipy.special import expit, logit

INPUT_DIR = Path("./data/processed")
OUTPUT_DIR = Path("./data/smoothed")
RESULTS_DIR = Path("./data/results")

# Injected from Phase 1 (01_hyper_tuning.py) results
GLOBAL_N = 250
GLOBAL_M = 50

# Observed segments shorter than this are too few points to filter; left NaN.
MIN_SEGMENT_ROWS = 10

# ==============================================================================
# 1. STATE-SPACE MODEL
# ==============================================================================
class HRVParticleModel(ssm.StateSpaceModel):
    """Bounded local-level + two circadian harmonics, in logit space.

    The observable is lo + (hi-lo)*sigmoid(level + c1 + c2), so every smoothed
    value is confined to (lo, hi) — encoding H1 (HRV is bounded and saturates at
    a ceiling). The latent states (level, harmonics) live in unbounded logit
    units; the sigmoid absorbs ceiling saturation instead of overshooting it.
    No deterministic slope (H4: drift is mild → a random-walk level suffices and
    cannot extrapolate). Gaussian observation (H6: first differences near-normal).
    """
    default_params = {
        'sigma_level': 0.05, 'sigma_seas': 0.01, 'sigma_obs': 10.0,
        'lo': 0.0, 'hi': 1.0,
        'z_level_loc': 0.0, 'z_level_scale': 1.0, 'z_seas_scale': 1.0,
        'omega1': 2 * np.pi / 144, 'omega2': 2 * np.pi / 72,
        'dt_norm': None,
    }

    def _observable(self, x):
        return self.lo + (self.hi - self.lo) * expit(x['level'] + x['c1'] + x['c2'])

    def PX0(self):
        return dists.StructDist({
            'level':   dists.Normal(loc=self.z_level_loc, scale=self.z_level_scale),
            'c1':      dists.Normal(loc=0.0, scale=self.z_seas_scale),
            'c1_star': dists.Normal(loc=0.0, scale=self.z_seas_scale),
            'c2':      dists.Normal(loc=0.0, scale=self.z_seas_scale / 2),
            'c2_star': dists.Normal(loc=0.0, scale=self.z_seas_scale / 2),
        })

    def PX(self, t, xp):
        dt = self.dt_norm[t] if self.dt_norm is not None else 1.0
        a1, b1 = np.cos(self.omega1 * dt), np.sin(self.omega1 * dt)
        a2, b2 = np.cos(self.omega2 * dt), np.sin(self.omega2 * dt)

        return dists.StructDist({
            'level':   dists.Normal(loc=xp['level'],                          scale=self.sigma_level * np.sqrt(dt)),
            'c1':      dists.Normal(loc=xp['c1'] * a1 + xp['c1_star'] * b1,   scale=self.sigma_seas),
            'c1_star': dists.Normal(loc=-xp['c1'] * b1 + xp['c1_star'] * a1,  scale=self.sigma_seas),
            'c2':      dists.Normal(loc=xp['c2'] * a2 + xp['c2_star'] * b2,   scale=self.sigma_seas),
            'c2_star': dists.Normal(loc=-xp['c2'] * b2 + xp['c2_star'] * a2,  scale=self.sigma_seas),
        })

    def PY(self, t, xp, x):
        return dists.Normal(loc=self._observable(x), scale=self.sigma_obs)


def compute_data_driven_params(y, median_dt_minutes, lo, hi):
    """Data-driven SSM parameters for one observed segment (y has no NaN).

    Spreads are measured in logit space because that is where the latent states
    live; sigma_obs stays in raw HRV units because the observation is raw HRV.
    """
    z = logit(np.clip((y - lo) / (hi - lo), 1e-6, 1 - 1e-6))
    n_per_day = max(int(24 * 60 / median_dt_minutes), 50)
    z_spread = max(float(1.4826 * np.median(np.abs(z - np.median(z)))), 0.2)

    sigma_obs = max(float(1.4826 * np.median(np.abs(np.diff(y))) / np.sqrt(2)), 1.0)
    obs_per_day = 24 * 60 / median_dt_minutes

    return {
        'lo': float(lo), 'hi': float(hi),
        'z_level_loc':   float(np.median(z[:n_per_day])),
        'z_level_scale': z_spread / 2,            # baseline: slow, narrow prior
        'z_seas_scale':  z_spread,                # harmonics carry the daily swing
        'sigma_obs':     sigma_obs,
        'sigma_level':   0.02 * z_spread,         # H4: slow baseline drift only
        'sigma_seas':    0.01,                    # circadian shape ~ stable day-to-day
        'omega1':        float(2 * np.pi / obs_per_day),
        'omega2':        float(2 * np.pi / (obs_per_day / 2)),
    }


# ==============================================================================
# 2. WORKER
# ==============================================================================
def _smooth_segment(y, dt_norm, median_dt, lo, hi, N_particles, M_paths):
    """Runs SMC + FFBS on one contiguous observed segment.

    Returns (smoothed_hrv, true_trend_level), both squashed back to HRV units
    and therefore inside (lo, hi):
    smoothed_hrv     = sigmoid(level + c1 + c2)  (baseline + 24h/12h harmonics)
    true_trend_level = sigmoid(level)            (baseline only, circadian removed)
    """
    params = compute_data_driven_params(y, median_dt, lo, hi)
    params['dt_norm'] = dt_norm

    fk = ssm.Bootstrap(ssm=HRVParticleModel(**params), data=y)
    alg = particles.SMC(fk=fk, N=N_particles, resampling='systematic',
                        store_history=True, verbose=False)
    alg.run()

    traj = alg.hist.backward_sampling_ON2(M_paths)
    squash = lambda z: lo + (hi - lo) * expit(z)
    observable = squash(np.array([x['level'] + x['c1'] + x['c2'] for x in traj]))
    level = squash(np.array([x['level'] for x in traj]))
    return observable.mean(axis=1), level.mean(axis=1)


def process_patient(args):
    file_path, output_dir, N_particles, M_paths = args
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        df.columns = df.columns.str.strip()

        if not {'createdTime', 'hrvValue', 'minute_diff'}.issubset(df.columns):
            return {'file': file_path.name, 'status': 'failed',
                    'reason': f'Missing columns. Found: {list(df.columns)}'}

        df['createdTime'] = pd.to_datetime(df['createdTime'])
        df = df.sort_values('createdTime').reset_index(drop=True)

        # The filter runs independently per contiguous observed segment; gap-fill
        # rows (gap_flag=1) stay NaN so nothing is interpolated across a
        # >GAP_THRESHOLD_MIN gap. Segment ids increment at each observed/missing edge.
        observed = df['hrvValue'].notna()
        df['gap_flag'] = (~observed).astype(int)
        df['smoothed_hrv'] = np.nan
        df['true_trend_level'] = np.nan
        seg_ids = (observed != observed.shift()).cumsum()

        # Sigmoid bounds = the patient's exact observed [min, max], shared across
        # all of this patient's segments. The open-interval sigmoid keeps every
        # smoothed value strictly inside (min, max); a mean asymptoting just under
        # the ceiling is the correct denoised estimate for saturated HRV.
        obs_vals = df.loc[observed, 'hrvValue'].to_numpy()
        lo, hi = float(obs_vals.min()), float(obs_vals.max())
        if hi - lo < 1.0:
            return {'file': file_path.name, 'status': 'failed',
                    'reason': 'observed range too narrow for bounded model'}

        n_segments = n_short = 0
        for _, seg in df[observed].groupby(seg_ids[observed]):
            if len(seg) < MIN_SEGMENT_ROWS:
                n_short += 1
                continue

            # Segment-local time steps: first row restarts at the nominal step
            # (its minute_diff is the preceding gap or NaN); guard duplicates.
            diffs = seg['minute_diff'].to_numpy(dtype=float)
            valid_diffs = diffs[1:][diffs[1:] > 0]
            median_dt = float(np.median(valid_diffs)) if len(valid_diffs) else 10.0
            dt_norm = diffs / median_dt
            dt_norm[0] = 1.0
            dt_norm[dt_norm <= 0] = 1.0

            smoothed, level = _smooth_segment(seg['hrvValue'].to_numpy(), dt_norm,
                                              median_dt, lo, hi, N_particles, M_paths)
            df.loc[seg.index, 'smoothed_hrv'] = smoothed
            df.loc[seg.index, 'true_trend_level'] = level
            n_segments += 1

        if n_segments == 0:
            return {'file': file_path.name, 'status': 'failed',
                    'reason': f'No observed segment >= {MIN_SEGMENT_ROWS} rows'}

        out_cols = ['createdTime', 'hrvValue', 'minute_diff',
                    'smoothed_hrv', 'true_trend_level', 'gap_flag']
        stem = file_path.stem.replace('_processed', '')
        df[out_cols].to_csv(Path(output_dir) / f"{stem}_smoothed.csv", index=False)

        return {'file': file_path.name, 'status': 'success', 'n_rows': len(df),
                'n_segments': n_segments, 'n_short_skipped': n_short}

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
