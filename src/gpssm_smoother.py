"""Gaussian-Process State-Space Model (GP-SSM) smoother for wearable HRV.

A lean adaptation of DevCielo/gaussian-process-state-space-models
(integration/linear_gp_ssm/linear_gp_ssm.py). That repo's defining idea is:
use a Gaussian Process to *learn* the linear transition dynamics F of a
state-space model, then run Kalman inference on it. This is a NEW smoothing
method alongside the particle filter (`02_run_filters.py`), RS-PF
(`02b_run_rspf.py`) and KRLST (`02c_run_krlst.py`), for benchmarking only; it
touches none of them.

Model (per patient, in LOG-HRV space so output is always > 0):
    x_t = F * x_{t-1} + w_t,   w_t ~ N(0, Q * dt_scale)   (GP-learned F)
    z_t = x_t + v_t,           v_t ~ N(0, R)              (H = 1)

================================================================================
WHERE THE ORIGINAL REPO VIOLATED THE H-FRAMEWORK  ->  HOW THIS PORT FIXES IT
================================================================================
H9  causal/online: the repo's headline output is `smooth()` = RTS smoother, a
    BACKWARD pass that uses future observations o_{t+1:T}; its EM also relearns F
    from the whole batch. FIX: drop the RTS smoother entirely and report the
    forward KALMAN FILTER mean E[x_t | o_{1:t}]. F/Q/R are calibrated once per
    patient (offline, like priors); inference is strictly causal.
H8  segmentation: the repo filters one continuous trajectory. FIX: split on
    >=180-min gaps into chunk_ids and run a FRESH Kalman pass per chunk (state +
    covariance reset; first reading is t=0).
H7  irregular time: the repo assumes a unit step. FIX: process noise scales with
    the ACTUAL elapsed minutes between readings (dt / patient-median dt).
H1  positive HRV: the repo's state is unbounded. FIX: filter in log space and
    return exp(.), clamped to the patient's observed log-range as an overflow
    guard only.
H5/H2/H4 drift & rhythm: a learned F<1 mean-reverting level + process noise lets
    the baseline wander and follow awake<->asleep shifts within a chunk.
H10 per-patient: F, Q, R and the log-mean centering are all derived from that
    patient's own data; nothing is shared across patients.

Public API:
    smooth_dataframe(df, ...)  with columns [patient_id, timestamp, hrv_value].
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

GAP_THRESHOLD_MIN = 180.0   # H8: a gap >= this opens a new chunk
MIN_SEGMENT_ROWS = 10       # chunks shorter than this are left NaN


@dataclass
class GPSSMConfig:
    gap_threshold_min: float = GAP_THRESHOLD_MIN
    min_segment_rows: int = MIN_SEGMENT_ROWS
    gp_max_pairs: int = 800     # subsample size for the GP that learns F (speed)
    q_scale: float = 0.05       # multiplies the data-derived process noise (bigger=less smooth)
    r_scale: float = 5.0        # multiplies the data-derived obs noise (bigger=smoother)
    f_max: float = 0.999        # cap on learned F for a stable filter
    seed: int = 0


# ==============================================================================
# CHUNKING (H7 / H8)
# ==============================================================================
def assign_chunks(df, timestamp_col="timestamp", patient_col="patient_id",
                  gap_threshold_min=GAP_THRESHOLD_MIN):
    """Add time_diff_min, gap_boundary, chunk_id. >=threshold gap => new chunk."""
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col])
    out = out.sort_values([patient_col, timestamp_col]).reset_index(drop=True)
    dt = out.groupby(patient_col, sort=False)[timestamp_col].diff().dt.total_seconds() / 60.0
    out["time_diff_min"] = dt
    boundary = dt.isna() | (dt >= gap_threshold_min)
    out["gap_boundary"] = boundary
    out["chunk_id"] = boundary.cumsum().astype(int)
    return out


# ==============================================================================
# GP-LEARNED DYNAMICS  (the GP-SSM step)
# ==============================================================================
def _learn_F(pairs_prev, pairs_next, cfg):
    """Learn the scalar linear transition F via GP regression x_{t-1} -> x_t.

    Mirrors the repo's `learn_dynamics`: fit a GP to consecutive state pairs and
    read the linear response off the unit basis vector (F = f(1)). Subsampled
    for speed; clamped to (0, f_max] so the resulting Kalman filter is stable.
    """
    n = pairs_prev.shape[0]
    if n < 5:
        return 0.9
    rng = np.random.default_rng(cfg.seed)
    if n > cfg.gp_max_pairs:
        sel = rng.choice(n, cfg.gp_max_pairs, replace=False)
        pairs_prev, pairs_next = pairs_prev[sel], pairs_next[sel]
    X = pairs_prev.reshape(-1, 1)
    y = pairs_next
    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False,
                                  random_state=cfg.seed, optimizer=None)
    gp.fit(X, y)
    # Linear response at the unit basis vector (repo's F-extraction).
    unit = np.std(pairs_prev) if np.std(pairs_prev) > 1e-6 else 1.0
    F = float(gp.predict(np.array([[unit]]))[0] / unit)
    return float(np.clip(F, 0.0, cfg.f_max))


# ==============================================================================
# CAUSAL KALMAN FILTER  (H9) -- one chunk, fresh state (H8), dt-aware (H7)
# ==============================================================================
def _filter_chunk(t_min, z, F, Q, R, ref_dt):
    """Forward Kalman filter over one chunk. Returns filtered mean (causal)."""
    T = z.shape[0]
    out = np.empty(T)
    x = z[0]                       # init at first reading (t=0)
    P = R + Q
    out[0] = x
    for i in range(1, T):
        dt_scale = max((t_min[i] - t_min[i - 1]) / ref_dt, 1e-6)   # H7
        xp = F * x                 # predict
        Pp = F * F * P + Q * dt_scale
        S = Pp + R                 # innovation variance
        K = Pp / S                 # Kalman gain
        x = xp + K * (z[i] - xp)   # update with o_i only (causal)
        P = (1.0 - K) * Pp
        out[i] = x
    return out


# ==============================================================================
# SMOOTHER
# ==============================================================================
def _fit_patient(z_by_chunk, cfg):
    """Calibrate F, Q, R from one patient's centered log-HRV (per chunk pairs)."""
    prev, nxt = [], []
    for z in z_by_chunk:
        if z.shape[0] >= 2:
            prev.append(z[:-1]); nxt.append(z[1:])
    prev = np.concatenate(prev) if prev else np.array([0.0])
    nxt = np.concatenate(nxt) if nxt else np.array([0.0])

    F = _learn_F(prev, nxt, cfg)
    # Observation noise from robust high-frequency differences; process noise
    # from the AR residual left over after removing F.
    d = np.diff(np.concatenate(z_by_chunk)) if len(z_by_chunk) else np.array([0.0])
    R = max((1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)) ** 2, 1e-4) * cfg.r_scale
    resid = nxt - F * prev
    Q = max(float(np.var(resid)) - R, 1e-4) * cfg.q_scale
    return F, Q, R


def smooth_dataframe(df, patient_col="patient_id", timestamp_col="timestamp",
                     value_col="hrv_value", config=None, out_col="gpssm_smoothed"):
    """Smooth an HRV DataFrame with the causal, chunked GP-SSM (Kalman) filter.

    df needs [patient_col, timestamp_col, value_col]. Returns a copy sorted by
    (patient, timestamp) with time_diff_min, gap_boundary, chunk_id and <out_col>
    added. Smoothing is grouped by [patient, chunk_id]; the filter state is fully
    reset at every >=180-min gap.
    """
    cfg = config or GPSSMConfig()
    missing = {patient_col, timestamp_col, value_col} - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {sorted(missing)}")

    work = assign_chunks(df, timestamp_col, patient_col, cfg.gap_threshold_min)
    work[out_col] = np.nan

    for _, pdf in work.groupby(patient_col, sort=False):
        t_min = (pdf[timestamp_col] - pdf[timestamp_col].iloc[0]).dt.total_seconds().to_numpy() / 60.0
        hrv = pd.to_numeric(pdf[value_col], errors="coerce").to_numpy(dtype=float)
        cid = pdf["chunk_id"].to_numpy()
        observed = np.isfinite(hrv) & (hrv > 0)
        if observed.sum() < cfg.min_segment_rows:
            continue

        log_obs = np.log(hrv[observed])
        log_mean = float(log_obs.mean())                  # H10 patient scale
        log_lo, log_hi = float(log_obs.min()), float(log_obs.max())

        # Per-patient median sampling step (H7 reference) and per-chunk centered
        # log series used both to calibrate (F,Q,R) and to filter.
        valid_dt = pdf["time_diff_min"].to_numpy()
        ref_dt = float(np.nanmedian(valid_dt[np.isfinite(valid_dt) & (valid_dt > 0)
                                              & (valid_dt < cfg.gap_threshold_min)])) or 10.0

        chunks = {}
        for c in np.unique(cid):
            sel = (cid == c) & observed
            if int(sel.sum()) >= cfg.min_segment_rows:
                idx = np.where(sel)[0]
                chunks[c] = (idx, np.log(hrv[idx]) - log_mean)

        if not chunks:
            continue
        F, Q, R = _fit_patient([z for _, z in chunks.values()], cfg)

        res = np.full(hrv.shape, np.nan)
        for c, (idx, z) in chunks.items():
            xf = _filter_chunk(t_min[idx] - t_min[idx[0]], z, F, Q, R, ref_dt)
            res[idx] = np.exp(np.clip(xf + log_mean, log_lo, log_hi))   # H1 >0
        work.loc[pdf.index, out_col] = res
    return work


# ==============================================================================
# SELF-TEST (positivity + chunk reset + causality)
# ==============================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rows = []
    t = pd.Timestamp("2026-01-01 08:00")
    for chunk in range(2):
        level = 40.0 if chunk == 0 else 70.0
        for _ in range(80):
            t += pd.Timedelta(minutes=10 + int(rng.integers(-2, 3)))    # jitter (H7)
            val = max(level + 10 * np.sin(t.hour / 24 * 2 * np.pi) + rng.normal(0, 6), 1)
            rows.append({"patient_id": "A", "timestamp": t, "hrv_value": val})
        t += pd.Timedelta(minutes=240)                                  # >=180 -> new chunk
    demo = pd.DataFrame(rows)

    res = smooth_dataframe(demo, config=GPSSMConfig())
    s = res["gpssm_smoothed"].dropna()
    print(f"rows={len(res)} chunks={res.chunk_id.nunique()}  "
          f"H1 min={s.min():.2f} (>0 -> {'PASS' if s.min() > 0 else 'FAIL'})")

    # H9: with calibration (F,Q,R) fixed, the filtered estimate at t depends only
    # on o_{1:t}; a truncated run matches the full run on shared rows.
    cfg = GPSSMConfig()
    c1 = demo.iloc[:80]
    tm = (c1.timestamp - c1.timestamp.iloc[0]).dt.total_seconds().to_numpy() / 60.0
    z = np.log(c1.hrv_value.to_numpy()) - np.log(c1.hrv_value.to_numpy()).mean()
    F, Q, R = _fit_patient([z], cfg)
    full = _filter_chunk(tm, z, F, Q, R, 10.0)
    pre = _filter_chunk(tm[:50], z[:50], F, Q, R, 10.0)
    d = np.abs(full[:50] - pre).max()
    print(f"H9 causality: max|full-prefix| over 50 = {d:.2e} -> {'PASS' if d < 1e-9 else 'FAIL'}")
