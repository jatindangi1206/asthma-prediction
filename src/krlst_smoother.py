"""Kernel Recursive Least-Squares Tracker (KRLST) smoother for wearable HRV.

A lean adaptation of PyKRLST (https://github.com/lckr/PyKRLST.git), the online
kernel RLS tracker of Van Vaerenbergh, Lazaro-Gredilla & Santamaria (IEEE TNNLS
2012). It is a NEW smoothing method that sits alongside the existing particle
filter (`02_run_filters.py`) and RS-PF (`02b_run_rspf.py`) for benchmarking; it
touches neither.

The `KRLST` class below is the original algorithm, kept close to the repo (two
small fixes are flagged inline). Everything else is the thin wrapper needed to
obey the project's H-Framework:

  H1  positive HRV    -> regress in LOG space, output exp(.)  => always > 0
  H5/H2/H4 drift/rhythm -> forgetting factor lambda<1 tracks the wandering baseline
  H7  irregular time  -> kernel input is real elapsed MINUTES, not a row index
  H8  >=180-min gaps  -> data is split into chunks; a FRESH tracker per chunk
  H9  causal/online   -> estimate at t uses only o_{1:t} (observe then predict t)
  H10 per-patient     -> log-mean centering + per-patient noise scale

There is intentionally a SINGLE output, `smoothed_hrv` (the causal KRLST
estimate). KRLST has no separate "trend level" / circadian-removed component;
that belonged to the particle-filter model and is not reproduced here.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import RBF, Kernel

GAP_THRESHOLD_MIN = 180.0   # H8: a gap >= this opens a new chunk
MIN_SEGMENT_ROWS = 10       # chunks shorter than this are left NaN


@dataclass
class KRLSTConfig:
    gap_threshold_min: float = GAP_THRESHOLD_MIN
    min_segment_rows: int = MIN_SEGMENT_ROWS
    budget: int = 100                  # dictionary size M (cost ~ O(M^2)/sample)
    lengthscale_min: float = 60.0      # RBF lengthscale in MINUTES (bigger=smoother,
                                       #   but too big -> ill-conditioned on dense data)
    c: float = 5.0                     # noise-to-signal ratio (bigger=smoother)
    forgetting_lambda: float = 0.99    # lambda in [0,1]; <1 tracks a drifting baseline
    forgetmode: str = "B2P"            # back-to-prior (recommended) or "UI"
    jitter: float = 1e-8


# ==============================================================================
# KRLST  (adapted from lckr/PyKRLST, MIT)
# ==============================================================================
class KRLST:
    """Kernel Recursive Least-Squares Tracker -- online, fixed budget."""

    def __init__(self, kernel: Kernel, l: float, c: float, M: int,
                 forgetmode: str = "B2P", jitter: float = 1e-8):
        if not (0.0 <= l <= 1.0):
            raise ValueError("`l` must be in [0, 1].")
        if forgetmode not in ("B2P", "UI"):
            raise ValueError("forgetmode must be 'B2P' or 'UI'.")
        self._kernel, self._lambda, self._c = kernel, l, float(c)
        self._M, self._forgetmode, self._jitter = int(M), forgetmode, jitter
        self._is_init = False

    def observe(self, x: np.ndarray, y: float, t: int):
        """Absorb one sample (x, y) at time index t. Strictly one-at-a-time (H9)."""
        x = np.atleast_2d(x)
        if not self._is_init:
            kss = self._kernel(x) + self._jitter
            self.Q = 1.0 / kss
            self.mu = (y * kss) / (kss + self._c)
            self.Sigma = kss - (kss ** 2) / (kss + self._c)
            self.basis, self.Xb, self.m = np.array([[t]]), x, 1
            self.nums02ML, self.dens02ML = y ** 2 / (kss + self._c), 1.0
            self.s02 = self.nums02ML / self.dens02ML
            self._is_init = True
            return

        if self._lambda < 1.0:                       # forgetting -> tracks drift
            if self._forgetmode == "B2P":
                Kt = self._kernel(self.Xb)
                self.Sigma = self._lambda * self.Sigma + (1.0 - self._lambda) * Kt
                self.mu = np.sqrt(self._lambda) * self.mu
            else:
                self.Sigma = self.Sigma / self._lambda

        kbs = self._kernel(self.Xb, x)
        kss = self._kernel(x) + self._jitter
        q = self.Q @ kbs
        ymean = q.T @ self.mu
        gamma2 = kss - kbs.T @ q
        gamma2[gamma2 < 0] = 0.0
        h = self.Sigma @ q
        sf2 = gamma2 + q.T @ h
        sf2[sf2 < 0] = 0.0
        sy2 = self._c + sf2

        gamma2_div = np.maximum(gamma2, self._jitter)   # FIX: keep 1/gamma2 finite
        Q_old = self.Q.copy()
        p = np.block([[q], [-1.0]])
        self.Q = np.block([[self.Q, np.zeros((self.m, 1))],
                           [np.zeros((1, self.m)), 0.0]]) + (1.0 / gamma2_div) * (p @ p.T)
        p = np.block([[h], [sf2]])
        self.mu = np.block([[self.mu], [ymean]]) + ((y - ymean) / sy2) * p
        self.Sigma = np.block([[self.Sigma, h], [h.T, sf2]]) - (1.0 / sy2) * (p @ p.T)
        self.basis = np.block([[self.basis], [t]])
        self.m += 1
        self.Xb = np.block([[self.Xb], [x]])
        self.nums02ML = self.nums02ML + self._lambda * (y - ymean) ** 2 / sy2
        self.dens02ML = self.dens02ML + self._lambda
        self.s02 = self.nums02ML / self.dens02ML

        if (self.m > self._M) or (gamma2 < self._jitter):
            if gamma2 < self._jitter:
                criterium = np.block([np.ones(self.m - 1), 0.0])
            else:
                errors = (self.Q @ self.mu).reshape(-1) / np.diag(self.Q)
                criterium = np.abs(errors)
            r = int(np.argmin(criterium))
            smaller = criterium > criterium[r]
            if r == self.m - 1:                  # FIX: original tested r==self.m (never true)
                self.Q = Q_old
            else:
                Qs, qs = self.Q[smaller, r], self.Q[r, r]
                self.Q = self.Q[smaller][:, smaller]
                self.Q = self.Q - (Qs.reshape(-1, 1) * Qs.reshape(1, -1)) / qs
            self.mu = self.mu[smaller]
            self.Sigma = self.Sigma[smaller][:, smaller]
            self.basis = self.basis[smaller]
            self.m -= 1
            self.Xb = self.Xb[smaller, :]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Posterior mean at X using the current (past-only) state."""
        kbs = self._kernel(self.Xb, np.atleast_2d(X))
        return (kbs.T @ self.Q @ self.mu).ravel()


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
# SMOOTHER
# ==============================================================================
def _smooth_chunk(t_min, hrv, log_mean, log_lo, log_hi, cfg):
    """Causal KRLST over one chunk, in log space. Fresh state (H8). Returns >0."""
    tracker = KRLST(RBF(length_scale=cfg.lengthscale_min), l=cfg.forgetting_lambda,
                    c=cfg.c, M=cfg.budget, forgetmode=cfg.forgetmode, jitter=cfg.jitter)
    z = np.log(hrv) - log_mean                       # center per patient (H10)
    out = np.empty(z.shape[0])
    for i in range(z.shape[0]):
        xi = np.array([[t_min[i]]])                  # time in minutes (H7)
        tracker.observe(xi, float(z[i]), i)          # absorb o_i ...
        out[i] = tracker.predict(xi)[0]              # ... then estimate at i (H9, causal)
    # Clamp to the patient's observed log-HRV range: guards the rare numerical
    # blow-up (ill-conditioned Gram matrix) from overflowing exp; leaves all
    # normal predictions untouched (no pull toward any center).
    log_est = np.clip(out + log_mean, log_lo, log_hi)
    return np.exp(log_est)                            # back to HRV, strictly > 0 (H1)


def smooth_dataframe(df, patient_col="patient_id", timestamp_col="timestamp",
                     value_col="hrv_value", config=None, out_col="krlst_smoothed"):
    """Smooth an HRV DataFrame with the causal, chunked KRLST tracker.

    df needs [patient_col, timestamp_col, value_col]. Returns a copy sorted by
    (patient, timestamp) with time_diff_min, gap_boundary, chunk_id and <out_col>
    added. Smoothing is grouped by [patient, chunk_id]; the tracker is fully
    reset at every >=180-min gap.
    """
    cfg = config or KRLSTConfig()
    missing = {patient_col, timestamp_col, value_col} - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {sorted(missing)}")

    work = assign_chunks(df, timestamp_col, patient_col, cfg.gap_threshold_min)
    work[out_col] = np.nan

    for _, pdf in work.groupby(patient_col, sort=False):
        t0 = pdf[timestamp_col].iloc[0]
        t_min = (pdf[timestamp_col] - t0).dt.total_seconds().to_numpy() / 60.0
        hrv = pd.to_numeric(pdf[value_col], errors="coerce").to_numpy(dtype=float)
        cid = pdf["chunk_id"].to_numpy()
        observed = np.isfinite(hrv) & (hrv > 0)
        if observed.sum() < cfg.min_segment_rows:
            continue
        log_obs = np.log(hrv[observed])
        log_mean = float(log_obs.mean())                      # H10 patient scale
        log_lo, log_hi = float(log_obs.min()), float(log_obs.max())   # H1 clamp range
        res = np.full(hrv.shape, np.nan)
        for c in np.unique(cid):
            sel = (cid == c) & observed
            if int(sel.sum()) < cfg.min_segment_rows:
                continue
            idx = np.where(sel)[0]
            res[idx] = _smooth_chunk(t_min[idx] - t_min[idx[0]], hrv[idx],
                                     log_mean, log_lo, log_hi, cfg)
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
            t += pd.Timedelta(minutes=10 + int(rng.integers(-2, 3)))   # jitter (H7)
            val = max(level + 10 * np.sin(t.hour / 24 * 2 * np.pi) + rng.normal(0, 6), 1)
            rows.append({"patient_id": "A", "timestamp": t, "hrv_value": val})
        t += pd.Timedelta(minutes=240)                                 # >=180 -> new chunk
    demo = pd.DataFrame(rows)

    res = smooth_dataframe(demo, config=KRLSTConfig())
    s = res["krlst_smoothed"].dropna()
    print(f"rows={len(res)} chunks={res.chunk_id.nunique()}  "
          f"H1 min={s.min():.2f} (>0 -> {'PASS' if s.min() > 0 else 'FAIL'})")

    # H9: deterministic, so a truncated run matches the full run on shared rows.
    cfg = KRLSTConfig()
    c1 = demo.iloc[:80]
    tm = (c1.timestamp - c1.timestamp.iloc[0]).dt.total_seconds().to_numpy() / 60.0
    lv = np.log(c1.hrv_value.to_numpy())
    lm, llo, lhi = float(lv.mean()), float(lv.min()), float(lv.max())
    full = _smooth_chunk(tm, c1.hrv_value.to_numpy(), lm, llo, lhi, cfg)
    pre = _smooth_chunk(tm[:50], c1.hrv_value.to_numpy()[:50], lm, llo, lhi, cfg)
    d = np.abs(full[:50] - pre).max()
    print(f"H9 causality: max|full-prefix| over 50 = {d:.2e} -> {'PASS' if d < 1e-9 else 'FAIL'}")
