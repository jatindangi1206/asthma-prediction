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
from wearable.cpd._contracts import series, positive, hit, excess

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
    t, z, chunk = series(time, values, priors)
    positive(cusum_k=p["cusum_k"], acf_window_min=p["acf_window_min"])
    if not 0 < p["coverage"] <= 1:
        raise ValueError("coverage must be in (0,1]")
    n = len(z)
    step = _step(t, priors)
    ok = np.isfinite(z)
    spike = np.where(ok, np.abs(z), np.nan)

    drift = np.full(n, np.nan)
    sp = sn = 0.0
    for i in range(n):
        if i and (chunk[i] != chunk[i - 1] or t[i] - t[i - 1] > 1.5 * step):
            sp = sn = 0.0
        if ok[i]:
            sp = max(0.0, sp + z[i] - p["cusum_k"])
            sn = max(0.0, sn - z[i] - p["cusum_k"])
            drift[i] = max(sp, sn)

    # consecutive pairs (no gap wider than 1.5 steps), summed over a trailing window
    pair = np.zeros(n)
    good = np.zeros(n, bool)
    good[1:] = ok[1:] & ok[:-1] & (np.diff(t) <= 1.5 * step) & (chunk[1:] == chunk[:-1])
    prod = np.where(good, np.r_[0.0, z[1:] * z[:-1]], 0.0)
    pair[good] = 1
    c1, c2, cn = (np.r_[0.0, np.cumsum(a)] for a in (prod, prod ** 2, pair))
    lo = np.searchsorted(t, t - p["acf_window_min"], side="right")
    chunk_start = np.maximum.accumulate(np.where(
        np.r_[True, chunk[1:] != chunk[:-1]], np.arange(n), 0)) if n else np.empty(0, int)
    lo = np.maximum(lo, chunk_start)
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
    out["scores"] = s
    th = {**THRESHOLDS, **(priors or {}).get("thresholds", {})}
    out["available"] = np.any(np.stack([np.isfinite(v) & np.isfinite(th[k])
                                          for k, v in s.items()]), axis=0)
    # lowest priority first, so a spike overwrites an autocorrelation flag, which
    # overwrites a drift — the same precedence as before
    for name, pheno in (("drift", PHENO_DRIFT), ("acf", PHENO_ACF), ("spike", PHENO_SPIKE)):
        x = s[name]
        fired = hit(x, th[name])
        out["anomaly"][fired] = 1
        out["phenotype"][fired] = pheno
        out["magnitude"][fired] = x[fired]
        out["severity"][fired] = excess(x, th[name])[fired]
    return out


# --- registry metadata (auto-discovered; see wearable/registry.py) ---
LABEL = "Kalman innovation CUSUM / autocorrelation"
PHENOTYPES = (
    'Kalman_Volatility_Spike',
    'Kalman_Gradual_Drift',
    'Kalman_Autocorrelation_Change',
)
