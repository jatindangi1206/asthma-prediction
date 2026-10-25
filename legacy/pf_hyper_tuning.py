import os
import time
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

# ==============================================================================
# 1. MATHEMATICAL MODEL DEFINITIONS 
# ==============================================================================
class HRVParticleModel(ssm.StateSpaceModel):
    """Bounded local-level + two circadian harmonics, in logit space.

    The observable is lo + (hi-lo)*sigmoid(level + c1 + c2), so every smoothed
    value is confined to (lo, hi) — encoding H1 (HRV is bounded and saturates at
    a ceiling). The latent states (level, harmonics) live in unbounded logit
    units; the sigmoid absorbs ceiling saturation instead of overshooting it.
    No deterministic slope (H4: drift is mild → a random-walk level suffices and
    cannot extrapolate). Gaussian observation (H6: first differences near-normal).

    Kept identical to the model in 02_run_filters.py.
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
# 2. ISOLATED WORKER 
# ==============================================================================
def process_single_patient(file_path):
    try:
        T_max = 500  
        
        if os.path.getsize(file_path) < 10:
            return {'file': str(file_path.name), 'status': 'failed', 'reason': 'Empty Spark partition'}

        df = pd.read_csv(file_path, encoding='utf-8-sig')
        df.columns = df.columns.str.strip()
        
        if 'createdTime' not in df.columns or 'hrvValue' not in df.columns:
            return {'file': str(file_path.name), 'status': 'failed', 'reason': f'Missing columns. Found: {list(df.columns)}'}

        df['createdTime'] = pd.to_datetime(df['createdTime'])
        df = df.sort_values('createdTime').reset_index(drop=True)
        
        # Longest contiguous OBSERVED segment (gap-fill NaN rows split segments),
        # matching the per-segment scheme in 02_run_filters.py — so y has no NaN.
        observed = df['hrvValue'].notna()
        df['segment_id'] = (observed != observed.shift()).cumsum()
        obs_df = df[observed]
        longest_seg = obs_df['segment_id'].value_counts().idxmax()
        df_sub = obs_df[obs_df['segment_id'] == longest_seg].iloc[:T_max].reset_index(drop=True)

        if len(df_sub) < 50:
            return {'file': str(file_path.name), 'status': 'failed', 'reason': 'Segment too short (<50 rows)'}

        # Sigmoid bounds = patient's exact observed [min, max], same as 02.
        obs_all = df.loc[observed, 'hrvValue'].to_numpy()
        lo, hi = float(obs_all.min()), float(obs_all.max())
        if hi - lo < 1.0:
            return {'file': str(file_path.name), 'status': 'failed', 'reason': 'observed range too narrow for bounded model'}

        y = df_sub['hrvValue'].values
        td = df_sub['createdTime'].diff().dt.total_seconds().fillna(0).values / 60.0
        median_dt = np.median(td[td > 0])
        dt_norm = td / median_dt
        dt_norm[0] = 1.0

        params = compute_data_driven_params(y, median_dt, lo, hi)
        params['dt_norm'] = dt_norm

        def _eval_N(N):
            ess_medians, ess_mins = [], []
            for i in range(5): 
                np.random.seed(42 + i)
                fk = ssm.Bootstrap(ssm=HRVParticleModel(**params), data=y)
                alg = particles.SMC(fk=fk, N=N, resampling='systematic', store_history=True, verbose=False)
                alg.run()
                ess_over_N = np.array([w.ESS for w in alg.hist.wgts]) / N
                ess_medians.append(np.median(ess_over_N))
                ess_mins.append(np.min(ess_over_N))
            return np.mean(ess_medians), np.mean(ess_mins)

        # --- FIX: Track Convergence ---
        N_star = 1000
        n_converged = False
        for N in [100, 250, 500, 1000]:
            med_ess, min_ess = _eval_N(N)
            if med_ess >= 0.5 and min_ess >= 0.1:
                N_star = N
                n_converged = True
                break

        np.random.seed(42)
        fk_m = ssm.Bootstrap(ssm=HRVParticleModel(**params), data=y)
        alg_m = particles.SMC(fk=fk_m, N=N_star, resampling='systematic', store_history=True, verbose=False)
        alg_m.run()

        def _eval_M(M):
            smoothed_means, within_sds = [], []
            for i in range(3): 
                np.random.seed(100+i)
                traj = alg_m.hist.backward_sampling_ON2(M)
                obs = np.zeros((M, len(y)))
                for t in range(len(y)):
                    z = traj[t]['level'] + traj[t]['c1'] + traj[t]['c2']
                    obs[:, t] = lo + (hi - lo) * expit(z)
                smoothed_means.append(np.mean(obs, axis=0))
                within_sds.append(np.std(obs, axis=0, ddof=1))
            emp_mc_sd = np.std(smoothed_means, axis=0, ddof=1)
            posterior_sd = np.mean(within_sds, axis=0)
            return np.median(emp_mc_sd / np.maximum(posterior_sd, 1e-9))

        # --- FIX: Track Convergence ---
        M_star = 200 
        m_converged = False
        for M in [25, 50, 100, 200]:
            ratio = _eval_M(M)
            if ratio < 0.10:
                M_star = M
                m_converged = True
                break
                
        return {
            'file': str(file_path.name), 'status': 'success', 
            'N_star': N_star, 'M_star': M_star,
            'N_converged': n_converged, 'M_converged': m_converged
        }
        
    except Exception as e:
        return {'file': str(file_path.name), 'status': 'failed', 'reason': str(e)}

# ==============================================================================
# 3. PARALLEL ORCHESTRATION 
# ==============================================================================
def run_parallel_cohort_sweep(base_directory):
    base_path = Path(base_directory)
    print(f"Scanning {base_path} for patient HRV files...")
    
    all_csvs = list(base_path.rglob('*.csv'))
    csv_files = [f for f in all_csvs if 'results' not in f.parts]

    if not csv_files:
        print("ERROR: No valid CSV files found. Please check your data directory.")
        return

    print(f"Found {len(csv_files)} patient files.")
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_patient, filepath): filepath for filepath in csv_files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(csv_files), desc="Tuning Cohort"):
            results.append(future.result())

    df_res = pd.DataFrame(results)
    results_dir = base_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = results_dir / "hyperparameter_log.csv"
    df_res.to_csv(log_path, index=False)
    
    successes = df_res[df_res['status'] == 'success']
    
    print("\n" + "="*60)
    print("COHORT-LEVEL GLOBAL HYPERPARAMETER RECOMMENDATIONS")
    print("="*60)
    
    if len(successes) > 0:
        global_N = int(np.percentile(successes['N_star'], 90))
        global_M = int(np.percentile(successes['M_star'], 90))
        
        allowed_N = [100, 250, 500, 1000]
        allowed_M = [25, 50, 100, 200]
        
        if global_N > 1000:
            print(f"⚠️ Warning: 90th percentile N ({global_N}) exceeds tested limit. Clamping to 1000.")
        if global_M > 200:
            print(f"⚠️ Warning: 90th percentile M ({global_M}) exceeds tested limit. Clamping to 200.")

        # --- FIX: Log Non-Converged Patients ---
        unconverged_n = len(successes[successes['N_converged'] == False])
        unconverged_m = len(successes[successes['M_converged'] == False])
        if unconverged_n > 0 or unconverged_m > 0:
            print(f"⚠️ Note: {unconverged_n} patients failed to converge on N, and {unconverged_m} failed on M.")
            print(f"   These were safely capped at the maximum limits.")

        final_N = next((n for n in allowed_N if n >= global_N), 1000)
        final_M = next((m for m in allowed_M if m >= global_M), 200)

        print(f"\n✅ Recommended Global N* (Particles): {final_N}")
        print(f"✅ Recommended Global M* (Paths):     {final_M}")
    else:
        print("❌ All patient processing failed. Check the log file for specific error reasons.")

if __name__ == "__main__":
    BASE_DATA_DIR = "./data" 
    run_parallel_cohort_sweep(BASE_DATA_DIR)