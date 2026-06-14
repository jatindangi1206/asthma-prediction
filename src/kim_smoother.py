"""Markov-Switching State-Space (Kim filter) smoother for wearable HRV.

Adapted from ljyflores/bayes-for-markov-switching, a Python implementation of a
2-regime Markov-switching autoregressive model fit by Bayesian Gibbs sampling
(Lim et al., 2020). That repo's `hamFilt` runs a Hamilton forward filter and then
BACKWARD-SAMPLES the regimes (FFBS) inside an MCMC loop. The user asked for the
KIM FILTER specifically: Kim's (1994) combined Hamilton + Kalman filter for a
Markov-switching *state-space* model -- a strictly FORWARD, causal recursion. So
we keep the regime-switching idea and the Hamilton likelihood machinery but
replace the non-causal backward sampling + Gibbs MCMC with the Kim filter.

This is a NEW smoothing method alongside the particle filter (`02_run_filters.py`),
RS-PF (`02b`), KRLST (`02c`), GP-SSM (`02d`) and OSSA (`02e`), for benchmarking
only; it touches none of them.

Model (per patient, in LOG-HRV space so output > 0), 2 regimes s_t in {0,1}:
    s_t ~ sticky Markov(P)                                     (H2/H4 awake<->asleep)
    x_t = mu_{s} + phi*(x_{t-1} - mu_{s}) + w,  w ~ N(0, Q_s)  (regime mean-reverting level)
    z_t = x_t + v,                              v ~ N(0, R_s)  (H = 1)
The Kim filter tracks, per regime, a Kalman (mean, var) plus a Hamilton regime
probability, runs the M*M Kalman branches each step, then COLLAPSES them back to
M with Kim's moment-matching approximation. The reported value is the
regime-mixed filtered level E[x_t | o_{1:t}].

================================================================================
WHERE THE ORIGINAL REPO VIOLATED THE H-FRAMEWORK  ->  HOW THIS PORT FIXES IT
================================================================================
H9  causal/online: the repo's `hamFilt` backward-samples regimes (uses the
    future) and the whole estimator is batch Gibbs MCMC. FIX: the Kim filter is
    forward-only -- regime probs and state come from o_{1:t} alone. Parameters
    are calibrated once per patient (offline, like priors). No backward pass,
    no sampling.
H8  segmentation: the repo fits one continuous AR series. FIX: per-chunk reset
    of regime probabilities AND each regime's Kalman (mean, covariance); fresh
    at every >=180-min gap, first reading = t=0.
H7  irregular time: the repo's AR(k) assumes a unit step. FIX: mean-reversion
    and process noise scale with the ACTUAL elapsed minutes (phi^(dt/ref),
    Q*dt/ref).
H1  positive HRV: filter in log space, output exp(.) clamped to the patient's
    observed log-range (overflow guard only).
H5/H2/H4 rhythm & drift: two regimes with different means reproduce the
    multimodal, awake<->asleep baseline; the sticky switching + slow level track
    the circadian wander.
H10 per-patient: regime means, variances and the log-mean centering all come
    from that patient's own data; nothing is shared across patients.

Public API:
    smooth_dataframe(df, ...)  with columns [patient_id, timestamp, hrv_value].
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

GAP_THRESHOLD_MIN = 180.0   # H8: a gap >= this opens a new chunk
MIN_SEGMENT_ROWS = 10       # chunks shorter than this are left NaN


@dataclass
class KimConfig:
    gap_threshold_min: float = GAP_THRESHOLD_MIN
    min_segment_rows: int = MIN_SEGMENT_ROWS
    phi: float = 0.9          # within-regime AR(1) retention toward the regime mean
    stickiness: float = 0.95  # regime self-transition prob (regimes persist)
    q_scale: float = 0.1      # process-noise multiplier (smaller => smoother)
    r_scale: float = 3.0      # obs-noise multiplier    (larger  => smoother)
    var_ratio: float = 1.0    # regime-1 noise variance relative to regime-0


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
# KIM FILTER  (Hamilton + Kalman + collapse) -- one chunk, forward only (H9)
# ==============================================================================
def _filter_chunk(t_min, z, params, ref_dt):
    """Causal Kim filter over one chunk. Returns the regime-mixed filtered level.

    Fresh regime probs + per-regime Kalman state here => this IS the H8 reset.
    """
    mu = params["mu"]            # (2,) regime means (centered log units)
    Q = params["Q"]             # (2,) regime process variances
    R = params["R"]             # (2,) regime obs variances
    phi = params["phi"]
    P = params["P"]             # (2,2) sticky Markov transition
    K = 2
    T = z.shape[0]
    out = np.empty(T)

    # ---- init (t = 0): stationary regime prob, Kalman seeded at first obs -----
    pr = _stationary(P)
    m = np.array([z[0], z[0]], dtype=float)        # per-regime filtered mean
    V = np.array([Q[0] + R[0], Q[1] + R[1]])       # per-regime filtered var
    out[0] = float(np.sum(pr * m))

    for t in range(1, T):
        dt = max((t_min[t] - t_min[t - 1]) / ref_dt, 1e-6)     # H7
        phi_dt = phi ** dt

        m_ij = np.zeros((K, K)); V_ij = np.zeros((K, K)); like = np.zeros((K, K))
        for i in range(K):           # came from regime i
            for j in range(K):       # now in regime j
                # Kalman predict under regime-j mean-reverting dynamics
                mp = mu[j] + phi_dt * (m[i] - mu[j])
                Vp = phi_dt * phi_dt * V[i] + Q[j] * dt
                # Kalman update with z_t (H=1, obs noise R_j)
                S = Vp + R[j]
                Kg = Vp / S
                m_ij[i, j] = mp + Kg * (z[t] - mp)
                V_ij[i, j] = (1.0 - Kg) * Vp
                like[i, j] = np.exp(-0.5 * (z[t] - mp) ** 2 / S) / np.sqrt(2 * np.pi * S)

        # ---- Hamilton step: joint Pr(s_{t-1}=i, s_t=j | o_{1:t}) ------------
        jp = (pr[:, None] * P) * like                 # (K,K) unnormalised
        s = jp.sum()
        jp = jp / s if s > 0 else np.full((K, K), 1.0 / (K * K))
        pr = jp.sum(axis=0)                            # marginal Pr(s_t=j|o_{1:t})

        # ---- Kim collapse: M*M branches -> M (moment matching) --------------
        for j in range(K):
            if pr[j] > 1e-300:
                w = jp[:, j] / pr[j]
                m[j] = np.sum(w * m_ij[:, j])
                V[j] = np.sum(w * (V_ij[:, j] + (m[j] - m_ij[:, j]) ** 2))
            else:
                m[j], V[j] = mu[j], Q[j] + R[j]

        out[t] = float(np.sum(pr * m))                 # regime-mixed filtered level
    return out


def _stationary(P):
    """Stationary distribution of a 2x2 sticky transition matrix."""
    # pi P = pi  ->  for 2-state: pi0 = (1-p11)/(2-p00-p11)
    p00, p11 = P[0, 0], P[1, 1]
    denom = (2.0 - p00 - p11)
    if denom <= 1e-9:
        return np.array([0.5, 0.5])
    pi0 = (1.0 - p11) / denom
    return np.array([pi0, 1.0 - pi0])


# ==============================================================================
# SMOOTHER
# ==============================================================================
def _fit_patient(z_all, cfg):
    """Calibrate regime means/variances and transition from one patient (H10)."""
    # Two regime levels: low / high quantiles of the centered log signal
    # (e.g. awake-low-HRV vs asleep-high-HRV) -> the H2/H4 multimodality.
    mu = np.quantile(z_all, [0.25, 0.75])
    d = np.diff(z_all)
    base_r = max((1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)) ** 2, 1e-4)
    R = np.array([base_r, base_r * cfg.var_ratio]) * cfg.r_scale
    Q = np.array([base_r, base_r * cfg.var_ratio]) * cfg.q_scale
    s = cfg.stickiness
    P = np.array([[s, 1 - s], [1 - s, s]])
    return {"mu": mu, "Q": Q, "R": R, "phi": cfg.phi, "P": P}


def smooth_dataframe(df, patient_col="patient_id", timestamp_col="timestamp",
                     value_col="hrv_value", config=None, out_col="kim_smoothed"):
    """Smooth an HRV DataFrame with the causal, chunked Kim filter.

    df needs [patient_col, timestamp_col, value_col]. Returns a copy sorted by
    (patient, timestamp) with time_diff_min, gap_boundary, chunk_id and <out_col>
    added. Smoothing is grouped by [patient, chunk_id]; regime probabilities and
    every regime's Kalman state are fully reset at each >=180-min gap.
    """
    cfg = config or KimConfig()
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
        log_mean = float(log_obs.mean())                   # H10 patient scale
        log_lo, log_hi = float(log_obs.min()), float(log_obs.max())
        params = _fit_patient(log_obs - log_mean, cfg)     # calibrate on centered log

        valid_dt = pdf["time_diff_min"].to_numpy()
        in_chunk = np.isfinite(valid_dt) & (valid_dt > 0) & (valid_dt < cfg.gap_threshold_min)
        ref_dt = float(np.nanmedian(valid_dt[in_chunk])) if in_chunk.any() else 10.0

        res = np.full(hrv.shape, np.nan)
        for c in np.unique(cid):
            sel = (cid == c) & observed
            if int(sel.sum()) < cfg.min_segment_rows:
                continue
            idx = np.where(sel)[0]
            z = np.log(hrv[idx]) - log_mean
            xf = _filter_chunk(t_min[idx] - t_min[idx[0]], z, params, ref_dt)
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
        for _ in range(100):
            t += pd.Timedelta(minutes=10 + int(rng.integers(-2, 3)))    # jitter (H7)
            val = max(level + 12 * np.sin(t.hour / 24 * 2 * np.pi) + rng.normal(0, 6), 1)
            rows.append({"patient_id": "A", "timestamp": t, "hrv_value": val})
        t += pd.Timedelta(minutes=240)                                  # >=180 -> new chunk
    demo = pd.DataFrame(rows)

    res = smooth_dataframe(demo, config=KimConfig())
    s = res["kim_smoothed"].dropna()
    print(f"rows={len(res)} chunks={res.chunk_id.nunique()}  "
          f"H1 min={s.min():.2f} (>0 -> {'PASS' if s.min() > 0 else 'FAIL'})")

    # H9: with calibration fixed, the Kim filter is deterministic & forward-only,
    # so a truncated run is byte-identical to the full run on shared rows.
    cfg = KimConfig()
    c1 = demo.iloc[:100]
    tm = (c1.timestamp - c1.timestamp.iloc[0]).dt.total_seconds().to_numpy() / 60.0
    z = np.log(c1.hrv_value.to_numpy()) - np.log(c1.hrv_value.to_numpy()).mean()
    pars = _fit_patient(z, cfg)
    full = _filter_chunk(tm, z, pars, 10.0)
    pre = _filter_chunk(tm[:60], z[:60], pars, 10.0)
    d = np.abs(full[:60] - pre).max()
    print(f"H9 causality: max|full-prefix| over 60 = {d:.2e} -> {'PASS' if d < 1e-9 else 'FAIL'}")
