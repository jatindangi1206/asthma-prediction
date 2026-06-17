"""H-RKF: Huber Robust Kalman Filter (log-domain) smoother for wearable HRV.

A Python port of the Huber robust Kalman filter from cvxgrp/
iteratively_saturated_kalman_filter (Boyd et al., "Iteratively Saturated Kalman
Filtering"). That repo replaces the Kalman update's quadratic measurement term
with a HUBER loss -- quadratic for small residuals, linear (saturated) for large
ones -- so measurement/process outliers are down-weighted instead of dragging
the estimate. It is a NEW smoothing method alongside the particle filter
(`02_run_filters.py`), RS-PF (`02b`), KRLST (`02c`), GP-SSM (`02d`), OSSA
(`02e`), Kim (`02f`), Gamma DGLM (`02g`) and Log-MHW (`02h`); it touches none.

WHY ROBUST + LOG-DOMAIN FOR HRV
-------------------------------
Wearable HRV has heavy-tailed artifacts (motion, poor contact) -- sudden spikes
or dropouts that a quadratic Kalman update chases. The Huber update treats those
as outliers and ignores them. Running in the LOG domain (z = log y, output
exp(.)) makes the model multiplicative and guarantees positivity (H1).

HUBER UPDATE
------------
The repo solves, at each step,
    min_x  (1/2)(x-a)' P^{-1} (x-a)  +  rho_huber( (z - F x)/sigma ; delta )
exactly with CVXPY. For a scalar HRV measurement this is solved here by IRLS
(iteratively reweighted least squares) -- the exact Huber solution in a few
matrix-vector iterations, matching the repo's "iteratively saturated" idea
without a CVXPY dependency:
    e = (z - F x)/sigma ;  w = 1 if |e|<=delta else delta/|e|   (Huber weight)
    (P^{-1} + w F'F/R) x = P^{-1} a + w F'z/R                    (reweighted normal eqn)
Outlier => |e| large => w -> 0 => the measurement is ignored and x stays at the
prediction a. Covariance: C = (P^{-1} + w F'F / R)^{-1} (robust information form,
as in the repo's `_huber_cov_update`).

H-FRAMEWORK
-----------
H1 log-domain exp output => > 0. H2/H4 discounted local level. H5 Fourier
circadian harmonics in the state. H7 G rotation and discount scale with real
elapsed minutes. H8 state reset per chunk at >=180-min gaps (first reading t=0).
H9 forward filtering (causal); the per-patient noise/threshold are offline
calibration like every sibling's priors. H10 measurement noise R and the prior
level are per patient. Missing y => predict only (no update).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

GAP_THRESHOLD_MIN = 180.0
MIN_SEGMENT_ROWS = 10
CIRCADIAN_PERIOD_MIN = 24 * 60.0


@dataclass
class HRKFConfig:
    gap_threshold_min: float = GAP_THRESHOLD_MIN
    min_segment_rows: int = MIN_SEGMENT_ROWS
    n_harmonics: int = 2
    period_min: float = CIRCADIAN_PERIOD_MIN
    deltrend: float = 0.99        # discount on the level (closer to 1 => smoother)
    delseas: float = 0.99         # discount on the circadian harmonics
    huber_delta: float = 1.345    # Huber threshold in std units (smaller => more robust)
    r_scale: float = 1.0          # multiplies the data-derived measurement variance
    n_irls: int = 5               # IRLS iterations for the Huber solve
    ref_dt_min: float = 10.0
    C0_var: float = 1.0


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


# ==============================================================================
# HUBER ROBUST KALMAN FILTER (log-domain)
# ==============================================================================
class HuberRobustKF:
    """Log-domain robust Kalman filter with a Huber measurement update (IRLS)."""

    def __init__(self, config: HRKFConfig | None = None):
        self.cfg = config or HRKFConfig()
        self._build()

    def _build(self):
        cfg = self.cfg
        H = cfg.n_harmonics
        self.p = 1 + 2 * H
        F = np.zeros(self.p); F[0] = 1.0
        disc = np.empty(self.p); disc[0] = cfg.deltrend
        self.omega = np.empty(H)
        for j in range(H):
            F[1 + 2 * j] = 1.0
            disc[1 + 2 * j] = disc[2 + 2 * j] = cfg.delseas
            self.omega[j] = 2 * np.pi * (j + 1) / cfg.period_min
        self.F, self.disc = F, disc

    def _G(self, dt):
        G = np.eye(self.p)
        for j in range(self.cfg.n_harmonics):
            a = self.omega[j] * dt
            ca, sa = np.cos(a), np.sin(a)
            i = 1 + 2 * j
            G[i, i] = ca; G[i, i + 1] = sa
            G[i + 1, i] = -sa; G[i + 1, i + 1] = ca
        return G

    def _disc_vec(self, dt):
        return self.disc ** (dt / self.cfg.ref_dt_min)

    def filter(self, t_min, z, m0, C0, R):
        """Forward causal Huber-robust filtering over one chunk (log signal z).

        Returns the filtered log-observable f_t = F'm_t (exp outside => HRV).
        Fresh m0/C0 here => the H8 reset. Missing z => predict only.
        """
        cfg = self.cfg
        F = self.F
        delta = cfg.huber_delta
        sigma = np.sqrt(R)
        T = z.shape[0]
        out = np.empty(T)
        m, C = m0.copy(), C0.copy()

        for t in range(T):
            dt = (t_min[t] - t_min[t - 1]) if t > 0 else cfg.ref_dt_min
            dt = max(dt, 1e-6)
            G = self._G(dt)
            a = G @ m
            P = G @ C @ G.T
            d = self._disc_vec(dt)
            P = P / np.outer(np.sqrt(d), np.sqrt(d))          # discount = process noise

            if np.isfinite(z[t]):
                Pinv = np.linalg.inv(P)
                x = a.copy()
                w = 1.0
                for _ in range(cfg.n_irls):                   # IRLS Huber solve
                    e = (z[t] - F @ x) / sigma
                    ae = abs(e)
                    w = 1.0 if ae <= delta else delta / ae    # Huber weight
                    A = Pinv + (w / R) * np.outer(F, F)
                    b = Pinv @ a + (w / R) * F * z[t]
                    x = np.linalg.solve(A, b)
                m = x
                C = np.linalg.inv(Pinv + (w / R) * np.outer(F, F))   # robust info update
            else:
                m, C = a, P                                   # missing => predict only

            out[t] = F @ m
        return out


# ==============================================================================
# SMOOTHER
# ==============================================================================
def smooth_dataframe(df, patient_col="patient_id", timestamp_col="timestamp",
                     value_col="hrv_value", config=None, out_col="hrkf_smoothed"):
    """Smooth an HRV DataFrame with the causal log-domain Huber robust KF."""
    cfg = config or HRKFConfig()
    missing = {patient_col, timestamp_col, value_col} - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {sorted(missing)}")

    work = assign_chunks(df, timestamp_col, patient_col, cfg.gap_threshold_min)
    work[out_col] = np.nan
    kf = HuberRobustKF(cfg)

    for _, pdf in work.groupby(patient_col, sort=False):
        t_min = (pdf[timestamp_col] - pdf[timestamp_col].iloc[0]).dt.total_seconds().to_numpy() / 60.0
        hrv = pd.to_numeric(pdf[value_col], errors="coerce").to_numpy(dtype=float)
        cid = pdf["chunk_id"].to_numpy()
        observed = np.isfinite(hrv) & (hrv > 0)
        if observed.sum() < cfg.min_segment_rows:
            continue

        logy = np.log(hrv[observed])
        log_med = float(np.median(logy))
        log_lo, log_hi = float(logy.min()), float(logy.max())
        # Measurement noise (log units) from robust first differences (H10).
        dz = np.diff(logy)
        R = max((1.4826 * np.median(np.abs(dz - np.median(dz))) / np.sqrt(2)) ** 2, 1e-4) * cfg.r_scale

        m0 = np.zeros(kf.p); m0[0] = log_med
        C0 = np.eye(kf.p) * cfg.C0_var

        res = np.full(hrv.shape, np.nan)
        for c in np.unique(cid):
            sel = (cid == c) & observed
            if int(sel.sum()) < cfg.min_segment_rows:
                continue
            idx = np.where(sel)[0]
            f = kf.filter(t_min[idx] - t_min[idx[0]], np.log(hrv[idx]), m0, C0, R)
            res[idx] = np.exp(np.clip(f, log_lo, log_hi))     # H1: > 0, within range
        work.loc[pdf.index, out_col] = res
    # Track B − Track A residual the CPD ensemble consumes (raw − smoothed).
    work["residual_hrv"] = pd.to_numeric(work[value_col], errors="coerce") - work[out_col]
    return work


# ==============================================================================
# SELF-TEST (positivity + chunk reset + causality + robustness)
# ==============================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rows = []
    t = pd.Timestamp("2026-01-01 08:00")
    for chunk in range(2):
        level = 40.0 if chunk == 0 else 70.0
        for k in range(150):
            t += pd.Timedelta(minutes=10 + int(rng.integers(-2, 3)))      # jitter (H7)
            mu = level * (1 + 0.2 * np.sin((t.hour * 60 + t.minute) / 1440 * 2 * np.pi))
            val = max(mu * np.exp(rng.normal(0, 0.1)), 1.0)
            if k % 25 == 7:
                val *= 3.0                                                # inject outlier spikes
            rows.append({"patient_id": "A", "timestamp": t, "hrv_value": val})
        t += pd.Timedelta(minutes=240)                                    # >=180 -> new chunk
    demo = pd.DataFrame(rows)

    res = smooth_dataframe(demo, config=HRKFConfig())
    s = res["hrkf_smoothed"].dropna()
    print(f"rows={len(res)} chunks={res.chunk_id.nunique()}  "
          f"H1 min={s.min():.2f} (>0 -> {'PASS' if s.min() > 0 else 'FAIL'})")

    # H9 causality: forward filter => truncated run matches full run on shared rows.
    cfg = HRKFConfig()
    kf = HuberRobustKF(cfg)
    c1 = demo.iloc[:150]
    tm = (c1.timestamp - c1.timestamp.iloc[0]).dt.total_seconds().to_numpy() / 60.0
    z = np.log(c1.hrv_value.to_numpy())
    m0 = np.zeros(kf.p); m0[0] = np.median(z); C0 = np.eye(kf.p)
    R = max((1.4826 * np.median(np.abs(np.diff(z))) / np.sqrt(2)) ** 2, 1e-4)
    full = kf.filter(tm, z, m0, C0, R)
    pre = kf.filter(tm[:90], z[:90], m0, C0, R)
    d = np.abs(full[:90] - pre).max()
    print(f"H9 causality: max|full-prefix| over 90 = {d:.2e} -> {'PASS' if d < 1e-9 else 'FAIL'}")

    # Robustness: Huber filter should be far less moved by the spikes than a plain
    # (delta=inf) Kalman filter on the same data.
    rob = smooth_dataframe(demo, config=HRKFConfig(huber_delta=1.0))["hrkf_smoothed"].to_numpy()
    plain = smooth_dataframe(demo, config=HRKFConfig(huber_delta=1e9))["hrkf_smoothed"].to_numpy()
    raw = demo.hrv_value.to_numpy()
    spike = (np.arange(len(raw)) % 25 == 7)
    print(f"Robustness at spikes: Huber RMSE-to-clean={np.sqrt(np.nanmean((rob[spike]-raw[spike]/3)**2)):.1f} "
          f"< plain={np.sqrt(np.nanmean((plain[spike]-raw[spike]/3)**2)):.1f} -> "
          f"{'PASS' if np.nanmean(np.abs(rob[spike]-raw[spike]/3)) < np.nanmean(np.abs(plain[spike]-raw[spike]/3)) else 'FAIL'}")
