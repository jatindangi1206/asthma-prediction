"""Online Singular Spectrum Analysis, adaptive window (OSSA) smoother for HRV.

A Python port + online adaptation of baggepinnen/SingularSpectrumAnalysis.jl.
That Julia package is BATCH SSA: build a Hankel trajectory matrix from the whole
series, SVD it, group singular components into trend/seasonal, and reconstruct
each group by diagonal averaging (`hankel` -> `hsvd` -> `elementary` ->
`unhankel`/`reconstruct`). This is a NEW smoothing method alongside the particle
filter (`02_run_filters.py`), RS-PF (`02b`), KRLST (`02c`) and GP-SSM (`02d`),
for benchmarking only; it touches none of them.

The SSA core kept from the repo:
    trajectory (Hankel) matrix  ->  SVD  ->  keep leading r components
    ->  low-rank reconstruction (the denoised signal).

================================================================================
WHERE BATCH SSA VIOLATED THE H-FRAMEWORK  ->  HOW THIS PORT FIXES IT
================================================================================
H9  causal/online: the repo reconstructs by DIAGONAL AVERAGING (`unhankel`),
    which averages each anti-diagonal of the trajectory matrix -- the value at
    time t mixes columns built from FUTURE samples o_{t+1:..}. Non-causal. FIX:
    the "online, adaptive window" scheme. At each step we take a trailing window
    ending at t, SVD it, and emit ONLY the newest reconstructed sample, i.e. the
    bottom-right element of the low-rank trajectory matrix,
        x_hat[t] = sum_{k<r} U[L-1,k] * S[k] * Vt[k, K-1].
    That anti-diagonal contains a single element (no future), so the estimate
    depends strictly on o_{1:t}.
H8  segmentation: batch SSA embeds across everything. FIX: process per chunk;
    the trailing window is reset at every >=180-min gap and never spans one. The
    first reading of a chunk is t=0.
ADAPTIVE WINDOW: within a chunk the window GROWS from the chunk start up to
    `window_max`, then slides. So the embedding adapts to how much causal history
    is available, and is flushed at gaps.
H7  irregular time: SSA is ordinal (lag embedding). We embed the actual
    consecutive readings in order WITHOUT resampling onto a grid (no
    interpolation), which is valid because within a chunk all gaps are <180 min.
H1  positive HRV: SSA runs in LOG space; output is exp(.), clamped to the
    patient's observed log-range as an overflow guard only.
H5/H2/H4 rhythm & drift: keeping the leading r components retains the trend and
    the dominant circadian/sub-circadian oscillations while discarding noise
    components -- exactly SSA's trend+seasonal grouping, done causally.
H10 per-patient: log-mean centering and the [lo,hi] clamp are per patient;
    nothing is shared across patients.

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
class OSSAConfig:
    gap_threshold_min: float = GAP_THRESHOLD_MIN
    min_segment_rows: int = MIN_SEGMENT_ROWS
    window_max: int = 144       # max trailing window (≈1 day at 10-min); adaptive up to this
    l_frac: float = 0.5         # lag-embedding dim L as a fraction of the window
    n_components: int = 2       # leading SVD components kept (smaller => smoother)
    min_window: int = 8         # below this, emit a causal trailing mean (warm-up)


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
# SSA CORE (ported from the Julia repo): trajectory matrix + low-rank last point
# ==============================================================================
def _hankel(y, L):
    """Trajectory (Hankel) matrix X[i, j] = y[i + j], shape (L, K), K = N-L+1.

    Same embedding as the repo's `hankel(y, L)`; built vectorized via a sliding
    window view (no copy of overlapping data until the SVD needs it).
    """
    N = y.shape[0]
    K = N - L + 1
    return np.lib.stride_tricks.sliding_window_view(y, L)[:K].T   # (L, K)


def _ossa_last(zwin, cfg):
    """Causal OSSA estimate of the NEWEST sample in the trailing window `zwin`.

    Builds the trajectory matrix, SVDs it, keeps the leading `n_components`, and
    returns the bottom-right element of the low-rank reconstruction -- the only
    anti-diagonal that contains no future sample (=> H9 causal).
    """
    n = zwin.shape[0]
    if n < cfg.min_window:
        return float(np.mean(zwin))                 # warm-up: causal trailing mean
    L = int(max(2, min(round(cfg.l_frac * n), n - 1)))
    X = _hankel(zwin, L)                            # (L, K)
    K = X.shape[1]
    # Economy SVD; keep r leading components (SSA trend + dominant oscillations).
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    r = int(min(cfg.n_components, S.shape[0]))
    # Newest reconstructed sample = X_r[L-1, K-1] = sum_k U[L-1,k] S[k] Vt[k,K-1].
    return float(np.sum(U[L - 1, :r] * S[:r] * Vt[:r, K - 1]))


# ==============================================================================
# SMOOTHER
# ==============================================================================
def _smooth_chunk(z, cfg):
    """Online adaptive-window SSA over one chunk (centered log signal).

    Window grows from the chunk start to `window_max`, then slides (adaptive).
    Fresh per chunk (H8). Returns the causal reconstructed level for every row.
    """
    T = z.shape[0]
    out = np.empty(T)
    W = cfg.window_max
    for i in range(T):
        zwin = z[max(0, i - W + 1): i + 1]          # trailing window ending at i
        out[i] = _ossa_last(zwin, cfg)              # emit newest sample only (H9)
    return out


def smooth_dataframe(df, patient_col="patient_id", timestamp_col="timestamp",
                     value_col="hrv_value", config=None, out_col="ossa_smoothed"):
    """Smooth an HRV DataFrame with online adaptive-window SSA.

    df needs [patient_col, timestamp_col, value_col]. Returns a copy sorted by
    (patient, timestamp) with time_diff_min, gap_boundary, chunk_id and <out_col>
    added. Smoothing is grouped by [patient, chunk_id]; the trailing window is
    fully reset at every >=180-min gap.
    """
    cfg = config or OSSAConfig()
    missing = {patient_col, timestamp_col, value_col} - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {sorted(missing)}")

    work = assign_chunks(df, timestamp_col, patient_col, cfg.gap_threshold_min)
    work[out_col] = np.nan

    for _, pdf in work.groupby(patient_col, sort=False):
        hrv = pd.to_numeric(pdf[value_col], errors="coerce").to_numpy(dtype=float)
        cid = pdf["chunk_id"].to_numpy()
        observed = np.isfinite(hrv) & (hrv > 0)
        if observed.sum() < cfg.min_segment_rows:
            continue

        log_obs = np.log(hrv[observed])
        log_mean = float(log_obs.mean())                   # H10 patient scale
        log_lo, log_hi = float(log_obs.min()), float(log_obs.max())

        res = np.full(hrv.shape, np.nan)
        for c in np.unique(cid):
            sel = (cid == c) & observed
            if int(sel.sum()) < cfg.min_segment_rows:
                continue
            idx = np.where(sel)[0]
            z = np.log(hrv[idx]) - log_mean                # centered log signal
            xf = _smooth_chunk(z, cfg)
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
        for _ in range(120):
            t += pd.Timedelta(minutes=10 + int(rng.integers(-2, 3)))    # jitter (H7)
            val = max(level + 12 * np.sin(t.hour / 24 * 2 * np.pi) + rng.normal(0, 6), 1)
            rows.append({"patient_id": "A", "timestamp": t, "hrv_value": val})
        t += pd.Timedelta(minutes=240)                                  # >=180 -> new chunk
    demo = pd.DataFrame(rows)

    res = smooth_dataframe(demo, config=OSSAConfig())
    s = res["ossa_smoothed"].dropna()
    print(f"rows={len(res)} chunks={res.chunk_id.nunique()}  "
          f"H1 min={s.min():.2f} (>0 -> {'PASS' if s.min() > 0 else 'FAIL'})")

    # H9: each output depends only on its trailing window, so a truncated run is
    # byte-identical to the full run on shared rows (deterministic SVD).
    cfg = OSSAConfig()
    c1 = demo.iloc[:120]
    z = np.log(c1.hrv_value.to_numpy()) - np.log(c1.hrv_value.to_numpy()).mean()
    full = _smooth_chunk(z, cfg)
    pre = _smooth_chunk(z[:70], cfg)
    d = np.abs(full[:70] - pre).max()
    print(f"H9 causality: max|full-prefix| over 70 = {d:.2e} -> {'PASS' if d < 1e-9 else 'FAIL'}")
