"""Log-MHW: Log-transformed Multiplicative Holt-Winters smoother for HRV.

Holt-Winters exponential smoothing (statsmodels.tsa.holtwinters.Exponential-
Smoothing) applied to log(HRV). Additive Holt-Winters on the log scale is a
MULTIPLICATIVE model on the original scale, and exp(.) of the result is always
positive -- so H1 is handled by the transform. This is a NEW smoothing method
for benchmarking alongside the particle filter (`02_run_filters.py`), RS-PF
(`02b`), KRLST (`02c`), GP-SSM (`02d`), OSSA (`02e`), Kim (`02f`) and the Gamma
DGLM (`02g`); it touches none of them.

HOLT-WINTERS RECURSIONS (additive, damped trend), on z_t = log y_t
------------------------------------------------------------------
    level    l_t = alpha (z_t - s_b) + (1-alpha)(l_{t-1} + phi*b_{t-1})
    trend    b_t = beta  (l_t - l_{t-1}) + (1-beta) phi*b_{t-1}
    season   s_b = gamma (z_t - l_t)     + (1-gamma) s_b        (b = bucket(t))
    output   y_hat_t = exp(l_t + s_b)                            (multiplicative)

statsmodels' ExponentialSmoothing is used (offline, per patient) to OPTIMISE the
smoothing parameters (alpha, beta, phi) by MLE on the patient's longest segment;
the causal recursion above then produces the online estimate.

H-FRAMEWORK ADAPTATION
----------------------
H1  positivity: log transform + exp output => strictly > 0 (intrinsic).
H5  circadian: the seasonal component is indexed by CLOCK time-of-day bucket
    (minute-of-day // 10, i.e. 144 buckets/day), seeded from a per-patient
    time-of-day profile and adapted online with gamma. Indexing by wall-clock
    time (not ordinal position) makes it robust to irregular sampling and gaps.
H7  irregular time: the damped trend decays with the ACTUAL elapsed minutes
    (phi^(dt/ref)); the seasonal bucket comes from the real timestamp. No grid
    resampling / interpolation.
H8  segmentation: level and trend are RESET at each >=180-min gap (first reading
    t=0); the seasonal vector is re-seeded from the patient profile per chunk.
H9  causal/online: the recursion is forward-only; y_hat_t uses o_{1:t}. The
    statsmodels parameter fit and the seasonal profile are offline per-patient
    calibration (like the priors in every sibling method).
H10 per-patient: the circadian profile and the smoothing parameters are derived
    from that patient's own data; nothing is shared across patients.
Missing y_t => carry the state forward (no level/trend/season update).
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

GAP_THRESHOLD_MIN = 180.0
MIN_SEGMENT_ROWS = 10
BUCKETS_PER_DAY = 144            # 10-min clock buckets (H5 / H7)


@dataclass
class LogMHWConfig:
    gap_threshold_min: float = GAP_THRESHOLD_MIN
    min_segment_rows: int = MIN_SEGMENT_ROWS
    alpha: float = 0.08          # level smoothing (smaller => smoother)
    beta: float = 0.02           # trend smoothing (small; HRV drift is mild)
    gamma: float = 0.02          # seasonal smoothing (profile is a strong seed)
    phi: float = 0.97            # trend damping (per reference step)
    ref_dt_min: float = 10.0     # reference sampling step for dt-scaling (H7)
    buckets_per_day: int = BUCKETS_PER_DAY
    optimize: bool = True        # optimise alpha/beta/phi via statsmodels per patient


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
# PER-PATIENT CALIBRATION (H10)
# ==============================================================================
def _tod_bucket(ts, buckets_per_day):
    """Clock time-of-day bucket index (0..buckets_per_day-1)."""
    minutes = ts.dt.hour * 60 + ts.dt.minute
    width = 1440.0 / buckets_per_day
    return np.minimum((minutes / width).astype(int).to_numpy(), buckets_per_day - 1)


def _seasonal_profile(z, buckets, m, smooth_win=9):
    """Per-patient circadian profile in log units, centered at 0 (H5).

    The per-bucket medians are then smoothed by a CIRCULAR moving average so the
    seasonal term is a clean daily shape rather than bucket-to-bucket noise
    (otherwise that noise would pass straight through to the output).
    """
    prof = np.zeros(m)
    for b in range(m):
        sel = buckets == b
        if sel.any():
            prof[b] = np.median(z[sel])
    filled = np.array([(buckets == b).any() for b in range(m)])
    idx = np.arange(m)
    if filled.any() and not filled.all():
        prof = np.interp(idx, idx[filled], prof[filled], period=m)
    # circular moving-average smoothing of the daily shape
    if smooth_win > 1:
        k = np.ones(smooth_win) / smooth_win
        prof = np.convolve(np.concatenate([prof[-smooth_win:], prof, prof[:smooth_win]]),
                           k, mode="same")[smooth_win:smooth_win + m]
    return prof - np.mean(prof)


def _optimize_params(z, cfg):
    """Optimise alpha/beta/phi via statsmodels Holt (damped) on a log segment."""
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(z, trend="add", damped_trend=True,
                                     seasonal=None, initialization_method="estimated")
        fit = model.fit(optimized=True)
        p = fit.params
        a = float(np.clip(p.get("smoothing_level", cfg.alpha), 0.01, 0.6))
        b = float(np.clip(p.get("smoothing_trend", cfg.beta), 0.0, 0.3))
        phi = float(np.clip(p.get("damping_trend", cfg.phi), 0.8, 0.999))
        return a, b, phi
    except Exception:
        return cfg.alpha, cfg.beta, cfg.phi


# ==============================================================================
# CAUSAL HOLT-WINTERS RECURSION (H9) -- one chunk, fresh level/trend (H8)
# ==============================================================================
def _filter_chunk(z, buckets, dt_scale, profile, params, ylo_log, yhi_log):
    """Forward additive Holt-Winters on log z. Returns exp(level+season) per row."""
    alpha, beta, gamma, phi = params
    s = profile.copy()                              # seasonal re-seeded per chunk (H8)
    T = z.shape[0]
    out = np.empty(T)

    b0 = buckets[0]
    level = z[0] - s[b0]                            # first reading -> t=0 (H8)
    trend = 0.0
    out[0] = np.exp(np.clip(level + s[b0], ylo_log, yhi_log))

    for t in range(1, T):
        phi_eff = phi ** dt_scale[t]                # damped trend over real dt (H7)
        bt = buckets[t]
        sb = s[bt]
        l_prev = level
        level = alpha * (z[t] - sb) + (1 - alpha) * (l_prev + phi_eff * trend)
        trend = beta * (level - l_prev) + (1 - beta) * phi_eff * trend
        s[bt] = gamma * (z[t] - level) + (1 - gamma) * sb
        out[t] = np.exp(np.clip(level + s[bt], ylo_log, yhi_log))   # H1: > 0
    return out


# ==============================================================================
# DATAFRAME API
# ==============================================================================
def smooth_dataframe(df, patient_col="patient_id", timestamp_col="timestamp",
                     value_col="hrv_value", config=None, out_col="logmhw_smoothed"):
    """Smooth an HRV DataFrame with the causal log multiplicative Holt-Winters."""
    cfg = config or LogMHWConfig()
    missing = {patient_col, timestamp_col, value_col} - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {sorted(missing)}")

    work = assign_chunks(df, timestamp_col, patient_col, cfg.gap_threshold_min)
    work[out_col] = np.nan
    m = cfg.buckets_per_day

    for _, pdf in work.groupby(patient_col, sort=False):
        ts = pdf[timestamp_col]
        hrv = pd.to_numeric(pdf[value_col], errors="coerce").to_numpy(dtype=float)
        cid = pdf["chunk_id"].to_numpy()
        observed = np.isfinite(hrv) & (hrv > 0)
        if observed.sum() < cfg.min_segment_rows:
            continue

        z_all = np.log(hrv)
        buckets_all = _tod_bucket(ts, m)
        ylo_log, yhi_log = float(np.log(hrv[observed].min())), float(np.log(hrv[observed].max()))
        profile = _seasonal_profile(z_all[observed], buckets_all[observed], m)   # H5/H10

        # Optimise alpha/beta/phi on the deseasonalised longest chunk (statsmodels).
        params = (cfg.alpha, cfg.beta, cfg.gamma, cfg.phi)
        if cfg.optimize:
            sizes = {c: int(((cid == c) & observed).sum()) for c in np.unique(cid)}
            cbig = max(sizes, key=sizes.get)
            if sizes[cbig] >= 30:
                sel = (cid == cbig) & observed
                zb = z_all[sel] - profile[buckets_all[sel]]
                a, b, phi = _optimize_params(zb, cfg)
                params = (a, b, cfg.gamma, phi)

        t_min = (ts - ts.iloc[0]).dt.total_seconds().to_numpy() / 60.0
        res = np.full(hrv.shape, np.nan)
        for c in np.unique(cid):
            sel = (cid == c) & observed
            if int(sel.sum()) < cfg.min_segment_rows:
                continue
            idx = np.where(sel)[0]
            tm = t_min[idx]
            dt_scale = np.ones(len(idx))
            dt_scale[1:] = np.maximum((tm[1:] - tm[:-1]) / cfg.ref_dt_min, 1e-6)
            res[idx] = _filter_chunk(z_all[idx], buckets_all[idx], dt_scale,
                                     profile, params, ylo_log, yhi_log)
        work.loc[pdf.index, out_col] = res
    # Track B − Track A residual the CPD ensemble consumes (raw − smoothed).
    work["residual_hrv"] = pd.to_numeric(work[value_col], errors="coerce") - work[out_col]
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
        for _ in range(200):
            t += pd.Timedelta(minutes=10 + int(rng.integers(-2, 3)))      # jitter (H7)
            mu = level * (1 + 0.2 * np.sin((t.hour * 60 + t.minute) / 1440 * 2 * np.pi))
            val = max(mu * np.exp(rng.normal(0, 0.12)), 1.0)              # multiplicative noise
            rows.append({"patient_id": "A", "timestamp": t, "hrv_value": val})
        t += pd.Timedelta(minutes=240)                                    # >=180 -> new chunk
    demo = pd.DataFrame(rows)

    res = smooth_dataframe(demo, config=LogMHWConfig(optimize=True))
    s = res["logmhw_smoothed"].dropna()
    print(f"rows={len(res)} chunks={res.chunk_id.nunique()}  "
          f"H1 min={s.min():.2f} (>0 -> {'PASS' if s.min() > 0 else 'FAIL'})")

    # H9 causality: forward recursion => truncated run matches full run (params and
    # profile held fixed as the per-patient calibration).
    cfg = LogMHWConfig(optimize=False)
    c1 = demo.iloc[:200].reset_index(drop=True)         # the first chunk by construction
    ts = c1.timestamp
    z = np.log(c1.hrv_value.to_numpy()); m = cfg.buckets_per_day
    bk = _tod_bucket(ts, m); prof = _seasonal_profile(z, bk, m)
    tm = (ts - ts.iloc[0]).dt.total_seconds().to_numpy() / 60.0
    dts = np.ones(len(z)); dts[1:] = np.maximum((tm[1:]-tm[:-1])/cfg.ref_dt_min, 1e-6)
    pr = (cfg.alpha, cfg.beta, cfg.gamma, cfg.phi)
    lo, hi = z.min(), z.max()
    full = _filter_chunk(z, bk, dts, prof, pr, lo, hi)
    pre = _filter_chunk(z[:120], bk[:120], dts[:120], prof, pr, lo, hi)
    d = np.abs(full[:120] - pre).max()
    print(f"H9 causality: max|full-prefix| over 120 = {d:.2e} -> {'PASS' if d < 1e-9 else 'FAIL'}")
