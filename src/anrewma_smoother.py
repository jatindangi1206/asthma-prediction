"""AN-REWMA: Adaptive Normalization + Causal Robust EWMA smoother for HRV.

A two-stage causal smoother:
  Stage 1 -- Adaptive Normalization: a causal rolling z-score that tracks a
             drifting baseline,  z_t = (y_t - mu_t) / sigma_t , with mu_t, sigma_t
             EWMA-estimated from PAST data only.
  Stage 2 -- Robust EWMA (WoLF): a Weighted-observation Likelihood Filter on the
             normalized signal, using an Inverse-Multi-Quadric (IMQ) weight to
             discount outliers.
  Stage 3 -- Back-transform:  y_hat_t = m_t * sigma_t + mu_t   (per-time mu/sigma).

This is the tenth NEW smoothing method for benchmarking, alongside the particle
filter (`02_run_filters.py`), RS-PF (`02b`), KRLST (`02c`), GP-SSM (`02d`), OSSA
(`02e`), Kim (`02f`), Gamma DGLM (`02g`), Log-MHW (`02h`) and H-RKF (`02i`). It
touches none of them.

EQUATIONS (WoLF EWMA, per step, on the normalized signal z_t)
------------------------------------------------------------
    w_t = (1 + (z_t - m_{t-1})^2 / c^2)^(-1/2)          (IMQ robustness weight)
    r_eff = r / w_t^2                                   (inflate noise for outliers)
    k_t = (s_{t-1} + q) / (s_{t-1} + q + r_eff)         (robust learning rate)
    m_t = k_t z_t + (1 - k_t) m_{t-1}                   (EWMA update)
    s_t = k_t r_eff
An outlier => large residual => small w_t => r_eff huge => k_t -> 0 => the point
is ignored (m_t stays put).

H-FRAMEWORK
-----------
H1  positivity: back-transform then clip into the patient's observed range (>0).
H2/H4 drift: the adaptive (rolling) mean mu_t follows the wandering baseline.
H5  rhythm: the rolling EWMA absorbs slow circadian variation.
H7  irregular time: the EWMA decay and process noise scale with the ACTUAL
    elapsed minutes (alpha_eff = 1-(1-alpha)^(dt/ref), q_eff = q*dt/ref).
H8  segmentation: both stages are RESET per chunk at >=180-min gaps (first
    reading t=0); nothing carries across a gap.
H9  causal/online: z_t uses only pre-update mu_t, sigma_t; WoLF is forward-only.
    Per-patient init is offline calibration (like every sibling's priors).
H10 per-patient: the normalization init and parameters are per patient.

Notes vs the reference sketch: the normalization stores PER-TIME mu_t, sigma_t
(so the back-transform is exact and causal), the rolling-variance recursion is
fixed, the EWMA update is made dt-aware, and the normalizer update is winsorized
so a single outlier cannot corrupt the rolling mean/std (extra robustness). No
numba dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

GAP_THRESHOLD_MIN = 180.0
MIN_SEGMENT_ROWS = 10


@dataclass
class ANREWMAConfig:
    gap_threshold_min: float = GAP_THRESHOLD_MIN
    min_segment_rows: int = MIN_SEGMENT_ROWS
    norm_alpha: float = 0.05      # EWMA rate for the rolling mean/std (smaller => smoother)
    norm_init_window: int = 20    # initial window for a stable z-score start
    winsor: float = 3.0           # clip |z| at this when UPDATING mu/sigma (robust norm)
    ewma_q: float = 0.02          # WoLF process noise (per reference step)
    ewma_r: float = 0.2           # WoLF measurement noise
    ewma_c: float = 1.0           # IMQ soft threshold in z units (smaller => more robust)
    ref_dt_min: float = 10.0      # reference sampling step for dt-scaling (H7)
    sigma_floor_frac: float = 0.05  # floor on sigma as a fraction of the init std


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


def _imq_weight(err, c):
    """Inverse-Multi-Quadric robustness weight in [0,1] (smaller for outliers)."""
    return 1.0 / np.sqrt(1.0 + (err * err) / (c * c))


# ==============================================================================
# AN-REWMA over ONE chunk (causal; fresh state => H8 reset)
# ==============================================================================
def _filter_chunk(y, dt_scale, cfg, ylo, yhi):
    """Adaptive normalization + WoLF robust EWMA + back-transform over one chunk."""
    n = y.shape[0]
    w0 = min(cfg.norm_init_window, n)
    mu = float(np.mean(y[:w0]))
    var = float(np.var(y[:w0]))
    sigma_floor = cfg.sigma_floor_frac * max(np.sqrt(var), 1e-6)
    sigma = max(np.sqrt(var), sigma_floor)

    m, s = 0.0, 1.0                      # WoLF state on the normalized signal
    out = np.empty(n)

    for t in range(n):
        yt = max(float(y[t]), 1e-6)
        z = (yt - mu) / sigma            # Stage 1: causal z-score (pre-update mu,sigma)

        # Stage 2: WoLF robust EWMA on z (dt-aware process noise, H7)
        q_eff = cfg.ewma_q * dt_scale[t]
        wt = _imq_weight(z - m, cfg.ewma_c)
        r_eff = cfg.ewma_r / (wt * wt)
        k = (s + q_eff) / (s + q_eff + r_eff)
        m = k * z + (1.0 - k) * m
        s = k * r_eff

        # Stage 3: back-transform with the SAME causal mu_t, sigma_t
        out[t] = np.clip(m * sigma + mu, ylo, yhi)   # H1: positive, in range

        # ---- update rolling mean/std (causal, dt-aware, winsorized) --------
        a = 1.0 - (1.0 - cfg.norm_alpha) ** dt_scale[t]     # H7 dt-aware alpha
        z_w = np.clip(z, -cfg.winsor, cfg.winsor)           # robust: cap outlier influence
        y_eff = mu + z_w * sigma
        mu = a * y_eff + (1.0 - a) * mu
        var = a * (y_eff - mu) ** 2 + (1.0 - a) * var
        sigma = max(np.sqrt(var), sigma_floor)
    return out


# ==============================================================================
# SMOOTHER
# ==============================================================================
def smooth_dataframe(df, patient_col="patient_id", timestamp_col="timestamp",
                     value_col="hrv_value", config=None, out_col="anrewma_smoothed"):
    """Smooth an HRV DataFrame with the causal AN-REWMA pipeline."""
    cfg = config or ANREWMAConfig()
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
        ylo, yhi = float(hrv[observed].min()), float(hrv[observed].max())   # H10 range

        res = np.full(hrv.shape, np.nan)
        for c in np.unique(cid):
            sel = (cid == c) & observed
            if int(sel.sum()) < cfg.min_segment_rows:
                continue
            idx = np.where(sel)[0]
            tm = t_min[idx]
            dt_scale = np.ones(len(idx))
            dt_scale[1:] = np.maximum((tm[1:] - tm[:-1]) / cfg.ref_dt_min, 1e-6)
            res[idx] = _filter_chunk(hrv[idx], dt_scale, cfg, ylo, yhi)     # H8 reset
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
                val *= 3.0                                                # outlier spikes
            rows.append({"patient_id": "A", "timestamp": t, "hrv_value": val})
        t += pd.Timedelta(minutes=240)                                    # >=180 -> new chunk
    demo = pd.DataFrame(rows)

    res = smooth_dataframe(demo, config=ANREWMAConfig())
    s = res["anrewma_smoothed"].dropna()
    print(f"rows={len(res)} chunks={res.chunk_id.nunique()}  "
          f"H1 min={s.min():.2f} (>0 -> {'PASS' if s.min() > 0 else 'FAIL'})")

    # H9 causality: forward + per-chunk init => truncated run matches the full run
    # on shared rows (init window common, so compare past the init window).
    cfg = ANREWMAConfig()
    c1 = demo.iloc[:150]
    tm = (c1.timestamp - c1.timestamp.iloc[0]).dt.total_seconds().to_numpy() / 60.0
    y = c1.hrv_value.to_numpy()
    ds = np.ones(len(y)); ds[1:] = np.maximum((tm[1:]-tm[:-1])/cfg.ref_dt_min, 1e-6)
    full = _filter_chunk(y, ds, cfg, y.min(), y.max())
    pre = _filter_chunk(y[:90], ds[:90], cfg, y.min(), y.max())
    d = np.abs(full[:90] - pre).max()
    print(f"H9 causality: max|full-prefix| over 90 = {d:.2e} -> {'PASS' if d < 1e-9 else 'FAIL'}")

    # Robustness: AN-REWMA vs a non-robust EWMA (huge IMQ threshold) at the spikes.
    rob = smooth_dataframe(demo, config=ANREWMAConfig(ewma_c=1.0))["anrewma_smoothed"].to_numpy()
    plain = smooth_dataframe(demo, config=ANREWMAConfig(ewma_c=1e6))["anrewma_smoothed"].to_numpy()
    raw = demo.hrv_value.to_numpy(); spike = (np.arange(len(raw)) % 25 == 7)
    er = np.nanmean(np.abs(rob[spike] - raw[spike] / 3)); ep = np.nanmean(np.abs(plain[spike] - raw[spike] / 3))
    print(f"Robustness at spikes: AN-REWMA dev={er:.1f} < non-robust dev={ep:.1f} -> "
          f"{'PASS' if er < ep else 'FAIL'}")
