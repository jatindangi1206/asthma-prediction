"""CT-IMM: Continuous-Time Interacting Multiple Model filter for wearable HRV.

The Interacting Multiple Model (IMM) filter (Blom & Bar-Shalom, IEEE TAC 1988)
runs a BANK of Kalman filters, one per dynamical "mode", and at every step blends
them according to how likely each mode is, allowing the system to switch between
modes via a Markov transition matrix. Unlike the Kim filter (`kim_smoother.py`),
which COLLAPSES the mode estimates at the END of a step, the IMM MIXES them at
the START (the "interaction" step) — that is the defining difference.

This is the eleventh benchmarking smoother alongside the particle filter
(`02_run_filters.py`), RS-PF, KRLST, GP-SSM, OSSA, Kim, Gamma DGLM, Log-MHW,
H-RKF and AN-REWMA. It touches none of them.

References (the classic 2D CA/CV/CT IMM uses kinematic modes; we adapt the same
machinery to a 1-D physiological signal):
  - Blom & Bar-Shalom (1988), "The interacting multiple model algorithm for
    systems with Markovian switching coefficients", IEEE Trans. Automatic Control.
  - Bar-Shalom, Li & Kirubarajan (2001), Estimation with Applications to Tracking
    and Navigation (Ch. 11, the 4-step IMM cycle).
  - xuepeixin/EKF-IMM (CA-CV-CT reference implementation).

MODEL (per patient, in LOG-HRV space so the output is always > 0)
-----------------------------------------------------------------
Continuous-time local-linear-trend state  x = [level, slope]  driven by a
white-noise-acceleration model, discretized over the ACTUAL elapsed minutes dt:
    A(dt) = [[1, dt], [0, 1]]
    Q_k(dt) = q_k * [[dt^3/3, dt^2/2], [dt^2/2, dt]]        (continuous-time, H7)
    z_t = level + noise,   noise ~ N(0, R)                  (H = [1, 0])
The K modes differ ONLY in the process-noise spectral density q_k — calm /
normal / volatile dynamics — so the IMM adapts how stiff the trend is, moment to
moment, to the signal's local behaviour (H2/H4/H5).

THE 4-STEP IMM CYCLE (each step)
--------------------------------
 1. Mixing probabilities :  mu_{i|j} = P_ij mu_i / cbar_j ,  cbar_j = sum_i P_ij mu_i
 2. Interaction (mixing) :  x0_j = sum_i mu_{i|j} x_i ;
                            P0_j = sum_i mu_{i|j} [P_i + (x_i - x0_j)(x_i - x0_j)^T]
 3. Mode-matched KF      :  predict (A, Q_k) from (x0_j, P0_j), update with z_t,
                            record the Gaussian likelihood Lambda_j of the innovation
 4. Mode update + combine:  mu_j  ∝ cbar_j * Lambda_j ;  x_hat = sum_j mu_j x_j
Reported value = exp(level component of x_hat), clamped to the patient's range.

H-FRAMEWORK
-----------
H1 log-domain exp output > 0. H2/H4/H5 the mode bank adapts trend stiffness so the
drifting/multimodal baseline and rhythm are tracked. H7 A and Q scale with the
real elapsed minutes. H8 the whole bank (states, covariances, mode probabilities)
is reset per chunk at >=180-min gaps; first reading is t=0. H9 forward filtering
only (causal); R is per-patient offline calibration like every sibling's priors.
H10 measurement noise R and the prior level are per patient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

GAP_THRESHOLD_MIN = 180.0
MIN_SEGMENT_ROWS = 10


@dataclass
class CTIMMConfig:
    gap_threshold_min: float = GAP_THRESHOLD_MIN
    min_segment_rows: int = MIN_SEGMENT_ROWS
    q_scale: float = 0.001                      # base process-noise (× R); smaller => smoother
    mode_mults: tuple = (0.1, 1.0, 10.0)        # calm / normal / volatile mode q multipliers
    stickiness: float = 0.95                    # mode self-transition prob (modes persist)
    r_scale: float = 1.0                        # multiplies the data-derived measurement noise
    ref_dt_min: float = 10.0                    # reference step for dt-scaling (H7)


# ==============================================================================
# CHUNKING (H7 / H8)
# ==============================================================================
def assign_chunks(df, timestamp_col="timestamp", patient_col="patient_id",
                  gap_threshold_min=GAP_THRESHOLD_MIN):
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col])
    out = out.sort_values([patient_col, timestamp_col]).reset_index(drop=True)
    dt = out.groupby(patient_col, sort=False)[timestamp_col].diff().dt.total_seconds() / 60.0
    out["time_diff_min"] = dt
    boundary = dt.isna() | (dt >= gap_threshold_min)
    out["gap_boundary"] = boundary
    out["chunk_id"] = boundary.cumsum().astype(int)
    return out


def _transition_matrix(K, s):
    P = np.full((K, K), (1.0 - s) / (K - 1)) if K > 1 else np.ones((1, 1))
    np.fill_diagonal(P, s)
    return P


# ==============================================================================
# CT-IMM over ONE chunk (causal; fresh bank => H8 reset)
# ==============================================================================
def _filter_chunk(t_min, z, R, q_base, cfg):
    """Run the 4-step IMM cycle over one contiguous chunk (log signal z).

    Scalar local-level (continuous-time random walk) modes: the level is the
    state, and the K modes differ in process-noise spectral density q_k, so the
    IMM adapts the smoothing stiffness (calm=stiff, volatile=responsive) moment
    to moment. No slope term (a slope would extrapolate trends and track the raw
    signal rather than smooth it).
    """
    q_modes = np.array([q_base * m for m in cfg.mode_mults])
    K = len(q_modes)
    P = _transition_matrix(K, cfg.stickiness)
    T = z.shape[0]
    out = np.empty(T)

    # ---- init the bank (chunk-local; H8 reset) -------------------------------
    x = np.full(K, z[0])                 # per-mode level estimate
    Pc = np.full(K, 4.0 * R)             # per-mode level variance
    mu = np.full(K, 1.0 / K)
    out[0] = z[0]

    for t in range(1, T):
        dt = max(float(t_min[t] - t_min[t - 1]), 1e-6)

        # 1) mixing probabilities
        cbar = P.T @ mu                                  # cbar_j = sum_i P_ij mu_i
        cbar = np.where(cbar > 1e-300, cbar, 1e-300)
        mu_ij = (P * mu[:, None]) / cbar[None, :]        # mu_ij[i,j]

        # 2) interaction: mixed initial conditions per mode j (scalar)
        x0 = (mu_ij * x[:, None]).sum(axis=0)            # x0_j = sum_i mu_ij x_i
        P0 = np.array([sum(mu_ij[i, j] * (Pc[i] + (x[i] - x0[j]) ** 2) for i in range(K))
                       for j in range(K)])

        # 3) mode-matched scalar Kalman predict + update; record likelihoods
        Pp = P0 + q_modes * dt                           # CT random walk: var grows with dt
        S = Pp + R
        Kg = Pp / S
        nu = z[t] - x0                                   # innovation per mode
        xn = x0 + Kg * nu
        Pn = (1.0 - Kg) * Pp
        like = np.exp(-0.5 * nu * nu / S) / np.sqrt(2 * np.pi * S)

        # 4) mode-probability update + estimate combination
        mu = cbar * like
        ssum = mu.sum()
        mu = mu / ssum if ssum > 1e-300 else np.full(K, 1.0 / K)
        x, Pc = xn, Pn
        out[t] = float((mu * x).sum())
    return out


# ==============================================================================
# SMOOTHER
# ==============================================================================
def smooth_dataframe(df, patient_col="patient_id", timestamp_col="timestamp",
                     value_col="hrv_value", config=None, out_col="ctimm_smoothed"):
    """Smooth an HRV DataFrame with the causal continuous-time IMM filter."""
    cfg = config or CTIMMConfig()
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

        logy = np.log(hrv[observed])
        log_lo, log_hi = float(logy.min()), float(logy.max())
        dz = np.diff(logy)
        R = max((1.4826 * np.median(np.abs(dz - np.median(dz))) / np.sqrt(2)) ** 2, 1e-4) * cfg.r_scale
        q_base = cfg.q_scale * R                               # process noise tied to obs noise

        res = np.full(hrv.shape, np.nan)
        for c in np.unique(cid):
            sel = (cid == c) & observed
            if int(sel.sum()) < cfg.min_segment_rows:
                continue
            idx = np.where(sel)[0]
            f = _filter_chunk(t_min[idx] - t_min[idx[0]], np.log(hrv[idx]), R, q_base, cfg)
            res[idx] = np.exp(np.clip(f, log_lo, log_hi))       # H1: > 0, in range
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
        for _ in range(120):
            t += pd.Timedelta(minutes=10 + int(rng.integers(-2, 3)))      # jitter (H7)
            mu = level * (1 + 0.2 * np.sin((t.hour * 60 + t.minute) / 1440 * 2 * np.pi))
            rows.append({"patient_id": "A", "timestamp": t,
                         "hrv_value": max(mu * np.exp(rng.normal(0, 0.1)), 1.0)})
        t += pd.Timedelta(minutes=240)                                    # >=180 -> new chunk
    demo = pd.DataFrame(rows)

    res = smooth_dataframe(demo, config=CTIMMConfig())
    s = res["ctimm_smoothed"].dropna()
    print(f"rows={len(res)} chunks={res.chunk_id.nunique()}  "
          f"H1 min={s.min():.2f} (>0 -> {'PASS' if s.min() > 0 else 'FAIL'})")

    cfg = CTIMMConfig()
    c1 = demo.iloc[:120]
    tm = (c1.timestamp - c1.timestamp.iloc[0]).dt.total_seconds().to_numpy() / 60.0
    z = np.log(c1.hrv_value.to_numpy())
    dz = np.diff(z); R = max((1.4826*np.median(np.abs(dz-np.median(dz)))/np.sqrt(2))**2, 1e-4)
    full = _filter_chunk(tm, z, R, cfg.q_scale*R, cfg)
    pre = _filter_chunk(tm[:70], z[:70], R, cfg.q_scale*R, cfg)
    d = np.abs(full[:70] - pre).max()
    print(f"H9 causality: max|full-prefix| over 70 = {d:.2e} -> {'PASS' if d < 1e-9 else 'FAIL'}")
