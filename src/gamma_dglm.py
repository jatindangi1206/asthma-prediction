"""Gamma Conjugate-Discount Dynamic Generalized Linear Model (CD-DGLM) for HRV.

A from-scratch Python implementation of the Gamma DGLM in the West & Harrison /
Triantafyllopoulos (arXiv:0802.0219) framework, in the style of PyBATS
(lavinei/pybats) and the CRAN SDGLM package. It is a NEW smoothing method for
benchmarking alongside the particle filter (`02_run_filters.py`), RS-PF (`02b`),
KRLST (`02c`), GP-SSM (`02d`), OSSA (`02e`) and the Kim filter (`02f`); it
touches none of them.

WHY A GAMMA DGLM FITS HRV
-------------------------
The Gamma observation model lives on the positive reals, so H1 (HRV > 0) is
satisfied STRUCTURALLY by the likelihood -- there is no log-transform-and-exp
trick or clamp as the other six methods need. Positivity is intrinsic.

MODEL
-----
    y_t | phi_t ~ Gamma(shape = s, rate = s * phi_t)      => E[y_t] = mu_t = 1/phi_t,
                                                              Var[y_t] = mu_t^2 / s
    lambda_t = log phi_t = -log mu_t = F' theta_t          (log link)
    theta_t  = G(dt) theta_{t-1} + omega_t                 (discounted evolution)

State theta = [ level , c1, c1*, c2, c2* ] : a drifting local level (H2/H4) plus
Fourier circadian harmonics (H5). F = [1, 1, 0, 1, 0].

SEQUENTIAL CONJUGATE UPDATING (West & Harrison guide map + Bayes linear)
-----------------------------------------------------------------------
Evolve:   a_t = G m_{t-1};  R_t = (G C_{t-1} G') ./ sqrt(d) sqrt(d)'  (block discount)
Predict:  f_t = F'a_t,  q_t = F'R_t F
Conjugate prior on phi_t matched to (f_t, q_t):  psi'(r_t) = q_t,  c_t = exp(psi(r_t) - f_t)
Update:   phi_t | y_t ~ Gamma(r_t + s, c_t + s y_t)
          f_t* = psi(r_t+s) - log(c_t + s y_t),   q_t* = psi'(r_t+s)
Correct:  m_t = a_t + R_t F (f_t* - f_t)/q_t
          C_t = R_t - R_t F F' R_t (1 - q_t*/q_t)/q_t
Filtered HRV (causal, H9):  E[mu_t | D_t] = (c_t + s y_t)/(r_t + s - 1).

FORECASTING
-----------
One-step / k-step forecast is the compound Gamma-Gamma:
    p(y) = Gamma(s+r)/(Gamma(s)Gamma(r)) * s^s c^r y^{s-1} / (c + s y)^{s+r},
    E[y] = c/(r-1).
`forecast` samples phi ~ Gamma(r,c) then y ~ Gamma(s, s phi). `log_likelihood`
sums the one-step predictive log-densities (Triantafyllopoulos model assessment).

H-FRAMEWORK
-----------
H1 positivity: intrinsic to the Gamma likelihood. H2/H4: discounted local level.
H5: Fourier harmonics. H7: G rotation angle and discount both scale with real
elapsed minutes. H8: prior is reset per chunk at >=180-min gaps (first reading
t=0). H9: `filter` is forward-only; the pipeline uses it. `smooth(mode='offline')`
is the retrospective (non-causal) DGLM smoother for offline analysis;
`smooth(mode='online')` returns the causal filtered estimate. H10: shape s and
the prior level are calibrated per patient. Missing y_t => evolve only (no update).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.special import digamma, polygamma, gammaln

GAP_THRESHOLD_MIN = 180.0
MIN_SEGMENT_ROWS = 10
CIRCADIAN_PERIOD_MIN = 24 * 60.0


@dataclass
class GammaDGLMConfig:
    gap_threshold_min: float = GAP_THRESHOLD_MIN
    min_segment_rows: int = MIN_SEGMENT_ROWS
    n_harmonics: int = 2          # Fourier circadian harmonics (H5)
    period_min: float = CIRCADIAN_PERIOD_MIN
    deltrend: float = 0.99        # discount on the level (smaller => adapts faster)
    delseas: float = 0.99         # discount on the seasonal harmonics
    shape: float | None = None    # Gamma obs shape s; if None, calibrated per patient
    shape_scale: float = 1.0      # multiplies the data-derived shape (bigger => smoother)
    q_floor: float = 1e-6
    q_cap: float = 5.0            # cap on prior linear-predictor variance (PyBATS safety)
    ref_dt_min: float = 10.0      # reference sampling step for dt-scaling (H7)
    C0_var: float = 1.0           # diffuse prior state variance


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
# GAMMA DGLM
# ==============================================================================
class GammaDGLM:
    """Gamma conjugate-discount DGLM with trend + circadian harmonics."""

    def __init__(self, config: GammaDGLMConfig | None = None):
        self.cfg = config or GammaDGLMConfig()
        self._build_components()
        self.shape = self.cfg.shape if self.cfg.shape is not None else 10.0
        self.m0 = np.zeros(self.p)
        self.C0 = np.eye(self.p) * self.cfg.C0_var
        self.loglik_ = np.nan
        self._store = None     # filter trace (for smoothing / forecasting)

    # -- components: F (regression vector), per-dim discount, harmonic freqs --
    def _build_components(self):
        cfg = self.cfg
        H = cfg.n_harmonics
        self.p = 1 + 2 * H
        F = np.zeros(self.p)
        F[0] = 1.0                                  # level
        disc = np.empty(self.p)
        disc[0] = cfg.deltrend
        self.omega = np.empty(H)
        for j in range(H):
            F[1 + 2 * j] = 1.0                       # cos term enters the predictor
            disc[1 + 2 * j] = disc[2 + 2 * j] = cfg.delseas
            self.omega[j] = 2 * np.pi * (j + 1) / cfg.period_min   # per-minute (H7)
        self.F = F
        self.disc = disc

    def _G(self, dt):
        """Evolution matrix: identity level + harmonic rotations by omega*dt (H7)."""
        G = np.eye(self.p)
        for j in range(self.cfg.n_harmonics):
            a = self.omega[j] * dt
            ca, sa = np.cos(a), np.sin(a)
            i = 1 + 2 * j
            G[i, i] = ca;  G[i, i + 1] = sa
            G[i + 1, i] = -sa; G[i + 1, i + 1] = ca
        return G

    def _disc_vec(self, dt):
        """Per-dim discount, scaled by elapsed time so longer gaps inject more variance."""
        return self.disc ** (dt / self.cfg.ref_dt_min)

    def _evolve(self, m, C, dt):
        G = self._G(dt)
        a = G @ m
        P = G @ C @ G.T
        d = self._disc_vec(dt)
        R = P / np.outer(np.sqrt(d), np.sqrt(d))    # block discounting
        return a, R, G

    @staticmethod
    def _conjugate_prior(f, q):
        """Solve psi'(r)=q for r, then c=exp(psi(r)-f). Gamma(r,c) prior on phi_t."""
        q = float(q)
        r = 1.0 / q if q > 0 else 1e3               # psi'(r) ~ 1/r for the initial guess
        for _ in range(60):                          # Newton on polygamma(1,r) - q
            g = polygamma(1, r) - q
            gp = polygamma(2, r)
            step = g / gp
            r_new = r - step
            if r_new <= 1e-6:
                r_new = r / 2.0
            if abs(r_new - r) < 1e-10:
                r = r_new; break
            r = r_new
        r = max(r, 1e-6)
        c = np.exp(digamma(r) - f)
        return r, c

    def _predict_lp(self, a, R):
        f = float(self.F @ a)
        q = float(self.F @ R @ self.F)
        q = min(max(q, self.cfg.q_floor), self.cfg.q_cap)
        return f, q

    @staticmethod
    def _compound_logpdf(y, s, r, c):
        """log p(y) for the Gamma-Gamma one-step predictive."""
        return (gammaln(s + r) - gammaln(s) - gammaln(r) + s * np.log(s)
                + r * np.log(c) + (s - 1) * np.log(y) - (s + r) * np.log(c + s * y))

    # -- per-patient calibration (H10) -------------------------------------
    def fit(self, t_min, y):
        """Calibrate the Gamma shape s and the prior level from one patient's data.

        s (observation precision) is set from the high-frequency coefficient of
        variation: Var[y]=mu^2/s  =>  s = mu^2 / noise_var, with noise_var from
        the robust MAD of first differences. The prior level is -log(median y).
        """
        y = np.asarray(y, dtype=float)
        obs = np.isfinite(y) & (y > 0)
        yv = y[obs]
        if yv.size < 3:
            raise ValueError("Need >=3 observations to fit.")
        if self.cfg.shape is not None:
            self.shape = float(self.cfg.shape)
        else:
            d = np.diff(yv)
            noise = (1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)) ** 2
            noise = max(noise, 1e-6)
            med = float(np.median(yv))
            self.shape = float(np.clip(med ** 2 / noise, 1.0, 1e4)) * self.cfg.shape_scale
        self.m0 = np.zeros(self.p)
        self.m0[0] = -np.log(float(np.median(yv)))     # level ~ -log mu
        self.C0 = np.eye(self.p) * self.cfg.C0_var
        self._ylo, self._yhi = float(yv.min()), float(yv.max())
        return self

    # -- forward causal filter (H9) ----------------------------------------
    def filter(self, t_min, y):
        """Forward conjugate-discount filtering over one contiguous chunk.

        Returns dict with filtered HRV mean `mu` (causal, E[mu_t|D_t]), one-step
        predictive mean `pred`, and the stored trace for smoothing/forecasting.
        Missing y => evolve only (no update). State starts at the prior (H8 reset).
        """
        t_min = np.asarray(t_min, dtype=float)
        y = np.asarray(y, dtype=float)
        T = y.shape[0]
        s = self.shape
        F = self.F
        ylo = getattr(self, "_ylo", np.nanmin(y[y > 0]) if np.any(y > 0) else 1.0)
        yhi = getattr(self, "_yhi", np.nanmax(y[y > 0]) if np.any(y > 0) else 1e3)
        # Positivity is intrinsic to the Gamma model; this clamp only tames the
        # rare diffuse-prior overshoot at a chunk's first points, and keeps the
        # output inside the patient's observed range (as the sibling smoothers do).
        clamp_lo, clamp_hi = ylo, yhi

        m, C = self.m0.copy(), self.C0.copy()
        mu = np.full(T, np.nan); pred = np.full(T, np.nan)
        a_s, R_s, m_s, C_s, G_s = [], [], [], [], []
        ll = 0.0

        for i in range(T):
            dt = (t_min[i] - t_min[i - 1]) if i > 0 else self.cfg.ref_dt_min
            dt = max(dt, 1e-6)
            a, R, G = self._evolve(m, C, dt)
            f, q = self._predict_lp(a, R)
            r, c = self._conjugate_prior(f, q)
            pred[i] = c / (r - 1) if r > 1 else np.nan          # prior forecast mean

            if np.isfinite(y[i]) and y[i] > 0:
                rstar, cstar = r + s, c + s * y[i]
                fstar = digamma(rstar) - np.log(cstar)
                qstar = polygamma(1, rstar)
                RF = R @ F
                m = a + RF * ((fstar - f) / q)
                C = R - np.outer(RF, RF) * ((1.0 - qstar / q) / q)
                mu[i] = np.clip(cstar / (rstar - 1), clamp_lo, clamp_hi)   # E[mu|D_t]
                ll += self._compound_logpdf(y[i], s, r, c)
            else:
                m, C = a, R                                     # missing => evolve only
                mu[i] = np.clip(c / (r - 1), clamp_lo, clamp_hi) if r > 1 else np.nan

            a_s.append(a); R_s.append(R); m_s.append(m.copy()); C_s.append(C.copy()); G_s.append(G)

        self.loglik_ = ll
        self.m, self.C = m, C                                   # final posterior (for forecast)
        self._store = dict(t=t_min, a=a_s, R=R_s, m=m_s, C=C_s, G=G_s, mu=mu, pred=pred)
        return dict(mu=mu, pred=pred, loglik=ll, m=m, C=C)

    # -- retrospective (offline) OR online smoothing -----------------------
    def smooth(self, t_min=None, y=None, mode="offline"):
        """Smoothed HRV mean.

        mode='online'  : the causal filtered estimate E[mu_t|D_t] (== filter; H9-safe).
        mode='offline' : the retrospective DGLM smoother E[mu_t|D_T], a backward
                         linear-Bayes pass over the filtered moments. NON-CAUSAL
                         (uses future observations) -- for offline analysis only.
        """
        if t_min is not None and y is not None:
            self.filter(t_min, y)
        if self._store is None:
            raise RuntimeError("Call filter() (or pass t_min,y) before smooth().")
        st = self._store
        if mode == "online":
            return st["mu"].copy()

        T = len(st["m"])
        ms = [None] * T; Cs = [None] * T
        ms[-1], Cs[-1] = st["m"][-1], st["C"][-1]
        F = self.F
        clo = getattr(self, "_ylo", 1e-3); chi = getattr(self, "_yhi", 1e6)
        out = np.full(T, np.nan)
        out[-1] = np.clip(np.exp(-float(F @ ms[-1])), clo, chi)
        for t in range(T - 2, -1, -1):
            G_next, R_next, a_next = st["G"][t + 1], st["R"][t + 1], st["a"][t + 1]
            B = st["C"][t] @ G_next.T @ np.linalg.inv(R_next)
            ms[t] = st["m"][t] + B @ (ms[t + 1] - a_next)
            Cs[t] = st["C"][t] + B @ (Cs[t + 1] - R_next) @ B.T
            f_s = float(F @ ms[t])
            out[t] = np.clip(np.exp(-f_s), clo, chi)           # mu = exp(-lambda)
        return out

    # -- multi-step forecast mean (deterministic path) ---------------------
    def predict(self, k, dt=None):
        """k-step-ahead forecast MEAN path E[y_{T+1:T+k}] from the current posterior."""
        if not hasattr(self, "m"):
            raise RuntimeError("Call filter() before predict().")
        dt = self.cfg.ref_dt_min if dt is None else dt
        a, R = self.m.copy(), self.C.copy()
        means = np.empty(k)
        for h in range(k):
            a, R, _ = self._evolve(a, R, dt)
            f, q = self._predict_lp(a, R)
            r, c = self._conjugate_prior(f, q)
            means[h] = c / (r - 1) if r > 1 else np.nan
        return means

    # -- forecast distribution (sampling) ----------------------------------
    def forecast(self, k, dt=None, nsamps=1000, quantiles=(0.05, 0.5, 0.95), seed=0):
        """Sample the k-step-ahead forecast distribution (compound Gamma-Gamma)."""
        if not hasattr(self, "m"):
            raise RuntimeError("Call filter() before forecast().")
        dt = self.cfg.ref_dt_min if dt is None else dt
        rng = np.random.default_rng(seed)
        s = self.shape
        a, R = self.m.copy(), self.C.copy()
        samples = np.empty((nsamps, k))
        for h in range(k):
            a, R, _ = self._evolve(a, R, dt)
            f, q = self._predict_lp(a, R)
            r, c = self._conjugate_prior(f, q)
            phi = rng.gamma(shape=r, scale=1.0 / c, size=nsamps)   # phi ~ Gamma(r,c)
            samples[:, h] = rng.gamma(shape=s, scale=1.0 / (s * phi))  # y|phi
        qs = np.quantile(samples, quantiles, axis=0)
        return dict(samples=samples, mean=samples.mean(axis=0),
                    quantiles={qq: qs[i] for i, qq in enumerate(quantiles)})

    # -- model assessment ---------------------------------------------------
    def log_likelihood(self, t_min=None, y=None):
        """Total one-step predictive log-likelihood (Triantafyllopoulos assessment)."""
        if t_min is not None and y is not None:
            self.filter(t_min, y)
        return self.loglik_


# ==============================================================================
# DATAFRAME API (causal filter per chunk)
# ==============================================================================
def smooth_dataframe(df, patient_col="patient_id", timestamp_col="timestamp",
                     value_col="hrv_value", config=None, out_col="gammadglm_smoothed"):
    """Smooth an HRV DataFrame with the causal Gamma CD-DGLM (filter per chunk)."""
    cfg = config or GammaDGLMConfig()
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

        model = GammaDGLM(cfg).fit(t_min[observed], hrv[observed])   # H10 calibration
        res = np.full(hrv.shape, np.nan)
        for c in np.unique(cid):
            sel = (cid == c) & observed
            if int(sel.sum()) < cfg.min_segment_rows:
                continue
            idx = np.where(sel)[0]
            out = model.filter(t_min[idx] - t_min[idx[0]], hrv[idx])   # H8 reset, H9 causal
            res[idx] = out["mu"]
        work.loc[pdf.index, out_col] = res
    return work


# ==============================================================================
# SELF-TEST (positivity + chunk reset + causality + forecast + loglik)
# ==============================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rows = []
    t = pd.Timestamp("2026-01-01 08:00")
    for chunk in range(2):
        level = 40.0 if chunk == 0 else 70.0
        for _ in range(120):
            t += pd.Timedelta(minutes=10 + int(rng.integers(-2, 3)))      # jitter (H7)
            mu = level + 12 * np.sin(t.hour / 24 * 2 * np.pi)
            val = max(rng.gamma(shape=20, scale=mu / 20), 1.0)            # gamma noise
            rows.append({"patient_id": "A", "timestamp": t, "hrv_value": val})
        t += pd.Timedelta(minutes=240)                                    # >=180 -> new chunk
    demo = pd.DataFrame(rows)

    res = smooth_dataframe(demo, config=GammaDGLMConfig())
    s = res["gammadglm_smoothed"].dropna()
    print(f"rows={len(res)} chunks={res.chunk_id.nunique()}  "
          f"H1 min={s.min():.2f} (>0 -> {'PASS' if s.min() > 0 else 'FAIL'})")

    # H9 causality: filter() is forward, so a truncated run matches the full run.
    cfg = GammaDGLMConfig()
    c1 = demo.iloc[:120]
    tm = (c1.timestamp - c1.timestamp.iloc[0]).dt.total_seconds().to_numpy() / 60.0
    yv = c1.hrv_value.to_numpy()
    mdl = GammaDGLM(cfg).fit(tm, yv)
    full = mdl.filter(tm, yv)["mu"]
    pre = GammaDGLM(cfg).fit(tm, yv).filter(tm[:70], yv[:70])["mu"]
    d = np.nanmax(np.abs(full[:70] - pre))
    print(f"H9 causality (filter): max|full-prefix| over 70 = {d:.2e} -> {'PASS' if d < 1e-9 else 'FAIL'}")

    # offline smoother, forecast, and log-likelihood sanity
    mdl.filter(tm, yv)
    sm_off = mdl.smooth(mode="offline"); sm_on = mdl.smooth(mode="online")
    fc = mdl.forecast(k=6, nsamps=2000)
    print(f"smooth offline>0: {'PASS' if np.nanmin(sm_off) > 0 else 'FAIL'}  "
          f"online==filter: {'PASS' if np.allclose(sm_on, mdl._store['mu'], equal_nan=True) else 'FAIL'}")
    print(f"forecast 6-step mean={np.round(fc['mean'],1)}  loglik={mdl.log_likelihood():.1f}")
    print(f"predict 6-step mean={np.round(mdl.predict(6),1)}")
