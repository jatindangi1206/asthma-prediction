"""Kalman innovation monitoring — volatility spikes, sustained drift, loss of whiteness.

The Kalman filter itself now lives in the noise model (`cpd/_whiten.py`): it turns the
residual into standardised innovations z, which are close to independent with unit
scale when nothing has changed. This detector watches those innovations for the three
innovation-based change types in the write-up:

  Kalman_Volatility_Spike       |z| is large                     stream "spike" = |z|
  Kalman_Gradual_Drift          a two-sided CUSUM of z breaches  stream "drift" = max(S+, S-)
                                (a slow one-directional creep)
  Kalman_Autocorrelation_Change the innovations stop being white stream "acf" = |sum z_i z_{i-1}|
                                over a trailing window in minutes       / sqrt(sum (z_i z_{i-1})^2)

`acf` is self-normalised, so it is ~|N(0,1)| on white noise whatever the tails. Each
stream's threshold comes from the detect step's calibration on no-change noise built
from the patient's own data (`priors["thresholds"]`); the fallbacks below are only for
standalone use. Causal: every statistic at t uses readings up to t.
"""

import numpy as np

DEFAULTS = {
    "cusum_k": 0.5,            # CUSUM slack, in z units (tuned to a ~1-sd shift)
    "acf_window_min": 120.0,   # trailing window for the whiteness check
    "coverage": 0.7,           # share of the window's expected readings it must hold
}
THRESHOLDS = {"spike": 4.0, "drift": 8.0, "acf": 3.5}   # fallbacks without calibration
PHENO_SPIKE = "Kalman_Volatility_Spike"
PHENO_DRIFT = "Kalman_Gradual_Drift"
PHENO_ACF = "Kalman_Autocorrelation_Change"


def _step(time, priors):
    s = (priors or {}).get("step_min")
    if s:
        return float(s)
    d = np.diff(np.asarray(time, float))
    return float(np.median(d)) if d.size else 1.0


def score(time, values, priors=None, **params):
    p = {**DEFAULTS, **params}
    t, z = np.asarray(time, float), np.asarray(values, float)
    n = len(z)
    step = _step(t, priors)
    ok = np.isfinite(z)
    spike = np.where(ok, np.abs(z), np.nan)

    drift = np.full(n, np.nan)
    sp = sn = 0.0
    for i in range(n):
        if ok[i]:
            sp = max(0.0, sp + z[i] - p["cusum_k"])
            sn = max(0.0, sn - z[i] - p["cusum_k"])
            drift[i] = max(sp, sn)

    # consecutive pairs (no gap wider than 1.5 steps), summed over a trailing window
    pair = np.zeros(n)
    good = np.zeros(n, bool)
    good[1:] = ok[1:] & ok[:-1] & (np.diff(t) <= 1.5 * step)
    prod = np.where(good, np.r_[0.0, z[1:] * z[:-1]], 0.0)
    pair[good] = 1
    c1, c2, cn = (np.r_[0.0, np.cumsum(a)] for a in (prod, prod ** 2, pair))
    lo = np.searchsorted(t, t - p["acf_window_min"], side="right")
    hi = np.arange(1, n + 1)
    m = cn[hi] - cn[lo]
    num, den = c1[hi] - c1[lo], c2[hi] - c2[lo]
    need = p["coverage"] * p["acf_window_min"] / step
    acf = np.where((m >= max(need, 5)) & (den > 0) & ok, np.abs(num) / np.sqrt(np.maximum(den, 1e-12)), np.nan)
    return {"spike": spike, "drift": drift, "acf": acf}


def detect(time, values, priors=None, **params):
    """Flag every reading where a stream is at or over its threshold."""
    n = len(values)
    out = {"anomaly": np.zeros(n, int),
           "phenotype": np.full(n, "normal", dtype=object),
           "magnitude": np.zeros(n), "severity": np.zeros(n)}
    if n < 2:
        return out
    s = score(time, values, priors, **params)
    th = {**THRESHOLDS, **(priors or {}).get("thresholds", {})}
    # lowest priority first, so a spike overwrites an autocorrelation flag, which
    # overwrites a drift — the same precedence as before
    for name, pheno in (("drift", PHENO_DRIFT), ("acf", PHENO_ACF), ("spike", PHENO_SPIKE)):
        x = np.nan_to_num(s[name], nan=0.0)
        hit = x >= th[name]
        out["anomaly"][hit] = 1
        out["phenotype"][hit] = pheno
        out["magnitude"][hit] = x[hit]
        out["severity"] = np.maximum(out["severity"], np.minimum(x / (2.0 * th[name]), 1.0))
    return out


# --- registry metadata (auto-discovered; see wearable/registry.py) ---
LABEL = "Kalman innovation CUSUM / autocorrelation"
PHENOTYPES = (
    'Kalman_Volatility_Spike',
    'Kalman_Gradual_Drift',
    'Kalman_Autocorrelation_Change',
)
