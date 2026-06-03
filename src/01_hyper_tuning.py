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

# ==============================================================================
# 1. MATHEMATICAL MODEL DEFINITIONS 
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
            'level': dists.Normal(loc=self.init_level_loc, scale=self.init_level_scale),
            'slope': dists.Normal(loc=0.0, scale=0.3),
            'c1': dists.Normal(loc=0.0, scale=35.0),     
            'c1_star': dists.Normal(loc=0.0, scale=35.0),
            'c2': dists.Normal(loc=0.0, scale=15.0),
            'c2_star': dists.Normal(loc=0.0, scale=15.0),
        })
        
    def PX(self, t, xp):
        dt = self.dt_norm[t] if self.dt_norm is not None else 1.0
        dt_sqrt = np.sqrt(dt)
        
        level_mean = xp['level'] + xp['slope'] * dt
        slope_mean = xp['slope']
        
        a1, b1 = np.cos(self.omega1 * dt), np.sin(self.omega1 * dt)
        a2, b2 = np.cos(self.omega2 * dt), np.sin(self.omega2 * dt)
        c1_m = xp['c1'] * a1 + xp['c1_star'] * b1
        c1_star_m = -xp['c1'] * b1 + xp['c1_star'] * a1
        c2_m = xp['c2'] * a2 + xp['c2_star'] * b2
        c2_star_m = -xp['c2'] * b2 + xp['c2_star'] * a2
        
        return dists.StructDist({
            'level': dists.Normal(loc=level_mean, scale=self.sigma_level * dt_sqrt),
            'slope': dists.Normal(loc=slope_mean, scale=self.sigma_slope * dt_sqrt),
            'c1': dists.Normal(loc=c1_m, scale=self.sigma_seas),
            'c1_star': dists.Normal(loc=c1_star_m, scale=self.sigma_seas),
            'c2': dists.Normal(loc=c2_m, scale=self.sigma_seas),
            'c2_star': dists.Normal(loc=c2_star_m, scale=self.sigma_seas),
        })
        
    def PY(self, t, xp, x):
        observable_mean = x['level'] + x['c1'] + x['c2']
        return dists.Student(df=self.nu, loc=observable_mean, scale=self.sigma_obs)

def compute_data_driven_params(y_data, mean_dt_minutes):
    y = y_data.astype(float)
    n_per_day = max(int(24 * 60 / mean_dt_minutes), 50)
    first_day_clean = y[:n_per_day][~np.isnan(y[:n_per_day])]
    
    if len(first_day_clean) > 5:
        init_level_loc = float(np.median(first_day_clean))
        init_level_scale = float(max(2 * np.median(np.abs(first_day_clean - init_level_loc)), 10.0))
    else:
        init_level_loc, init_level_scale = float(np.nanmedian(y)), 20.0
        
    diffs = np.diff(y)
    diffs = diffs[~np.isnan(diffs)]
    mad_diffs = np.median(np.abs(diffs - np.median(diffs))) if len(diffs) > 0 else 5.0
    
    sigma_obs = max(float(1.4826 * mad_diffs / np.sqrt(2)), 3.0)
    sigma_level = 0.05 * sigma_obs
    sigma_slope = 0.02 * sigma_level
    
    obs_per_day = 24 * 60 / mean_dt_minutes
    
    return {
        'init_level_loc': init_level_loc, 'init_level_scale': init_level_scale,
        'sigma_obs': sigma_obs, 'sigma_level': sigma_level, 'sigma_slope': sigma_slope,
        'sigma_seas': 0.15, # FIX: Explicitly setting seasonal drift here 
        'omega1': float(2 * np.pi / obs_per_day),
        'omega2': float(2 * np.pi / (obs_per_day / 2))
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
        
        df['is_new_segment'] = (df['createdTime'].diff() > pd.Timedelta(minutes=180)).astype(int)
        df['segment_id'] = df['is_new_segment'].cumsum()
        longest_seg = df.groupby('segment_id').size().idxmax()
        df_sub = df[df['segment_id'] == longest_seg].iloc[:T_max].reset_index(drop=True)
        
        if len(df_sub) < 50:
            return {'file': str(file_path.name), 'status': 'failed', 'reason': 'Segment too short (<50 rows)'}

        y = df_sub['hrvValue'].values
        td = df_sub['createdTime'].diff().dt.total_seconds().fillna(0).values / 60.0
        median_dt = np.median(td[td > 0])
        dt_norm = td / median_dt
        dt_norm[0] = 1.0

        params = compute_data_driven_params(y, median_dt)
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
                    obs[:, t] = traj[t]['level'] + traj[t]['c1'] + traj[t]['c2']
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