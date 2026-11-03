"""Bayesian Online Change Point Detection (Adams & MacKay, 2007) on the whitened residual.

Each run-length hypothesis carries a Normal–Gamma posterior over the segment's mean AND
variance, so its predictive is a Student-t: a change in spread is modelled (not just
labelled afterwards), and a heavy-tailed burst does not by itself collapse the run.

Time-aware hazard: changes are taken to arrive as a Poisson process in clock time, so
the probability that one happened since the previous reading is 1 - exp(-dt/lambda).
A long gap therefore raises the prior of a change on its own, a short one barely.

A new segment's first reading is scored under the PRIOR predictive. (With the textbook
recursion, which scores it under the old runs' predictive, P(r_t = 0) equals the hazard
exactly at every step and carries no information.) The statistic is

    cp_t = P(the current run started within the last `window_min` minutes | z up to t)

on the log-odds scale (probabilities pile up at 0.9999...), thresholded at the value
calibrated on no-change noise from the patient's own data.
Runs are kept in log space and pruned (relative mass < 1e-6, at most `max_hyp`), so
cost is linear in the series and nothing underflows.

On a flagged run's first reading the change is classified causally: compare the
`window_min` before it with the `window_min` ending at it — a larger mean shift is
BOCPD_Mean_Shift, a larger spread change BOCPD_Variance_Change.
"""

import numpy as np
from scipy.special import gammaln

DEFAULTS = {
    "lambda_min": 4320.0,      # expected clock time between changes (3 days)
    "window_min": 60.0,        # "recent" run: started within this many minutes
    "max_hyp": 300,            # run-length hypotheses kept after pruning
}
THRESHOLDS = {"cp": 0.0}       # log-odds; fallback without calibration (P = 0.5)
PHENO_MEAN = "BOCPD_Mean_Shift"
PHENO_VAR = "BOCPD_Variance_Change"
# Normal–Gamma prior on z (unit scale when the noise model is right): mean 0,
# E[variance] = beta/(alpha-1) = 1, weakly held.
_MU0, _K0, _A0, _B0 = 0.0, 1.0, 2.0, 1.0
_PRUNE = np.log(1e-6)


def _logt(x, mu, k, a, b):
    """Student-t log predictive of x under Normal–Gamma (mu, kappa, alpha, beta)."""
    var = b * (k + 1) / (a * k)
    nu = 2 * a
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi * var)
            - (nu + 1) / 2 * np.log1p((x - mu) ** 2 / (nu * var)))


def _post(x, mu, k, a, b):
    return ((k * mu + x) / (k + 1), k + 1, a + 0.5, b + k * (x - mu) ** 2 / (2 * (k + 1)))


def score(time, values, priors=None, **params):
    p = {**DEFAULTS, **params}
    t, z = np.asarray(time, float), np.asarray(values, float)
    cp = np.full(len(z), np.nan)
    lp = np.empty(0); start = np.empty(0)
    mu = k = a = b = np.empty(0)
    prev_t = None
    for i in range(len(z)):
        x = z[i]
        if not np.isfinite(x):
            continue
        if lp.size == 0:                                  # the very first reading
            mu, k, a, b = (np.array([v]) for v in _post(x, _MU0, _K0, _A0, _B0))
            lp, start = np.array([0.0]), np.array([t[i]])
            prev_t = t0 = t[i]
            continue
        h = -np.expm1(-(t[i] - prev_t) / p["lambda_min"])
        h = min(max(h, 1e-12), 1 - 1e-12)
        pred = _logt(x, mu, k, a, b)
        grow = lp + np.log1p(-h) + pred
        top = lp.max()
        new = np.log(h) + top + np.log(np.exp(lp - top).sum()) + _logt(x, _MU0, _K0, _A0, _B0)
        lp = np.r_[new, grow]
        start = np.r_[t[i], start]
        m2, k2, a2, b2 = _post(x, mu, k, a, b)
        m0, k0, a0, b0 = _post(x, _MU0, _K0, _A0, _B0)
        mu, k, a, b = np.r_[m0, m2], np.r_[k0, k2], np.r_[a0, a2], np.r_[b0, b2]
        top = lp.max()
        lp = lp - (top + np.log(np.exp(lp - top).sum()))
        keep = lp >= max(_PRUNE, np.sort(lp)[-int(p["max_hyp"])] if lp.size > p["max_hyp"] else _PRUNE)
        if not keep.all():
            lp, start, mu, k, a, b = lp[keep], start[keep], mu[keep], k[keep], a[keep], b[keep]
            top = lp.max()
            lp = lp - (top + np.log(np.exp(lp - top).sum()))
        # in the first `window_min` every run is "recent" by construction: no score yet
        if t[i] - t0 > p["window_min"]:
            q = min(max(float(np.exp(lp[start > t[i] - p["window_min"]]).sum()), 1e-15), 1 - 1e-15)
            cp[i] = np.log(q) - np.log1p(-q)          # log-odds: calibratable near 1
        prev_t = t[i]
    return {"cp": cp}


def detect(time, values, priors=None, **params):
    """Flag readings where a recent change is probable; classify each flagged run."""
    p = {**DEFAULTS, **params}
    t, z = np.asarray(time, float), np.asarray(values, float)
    N = len(z)
    out = {"anomaly": np.zeros(N, int),
           "phenotype": np.full(N, "normal", dtype=object),
           "magnitude": np.zeros(N), "severity": np.zeros(N)}
    if N < 2:
        return out
    lo = score(t, z, priors, **params)["cp"]
    th = {**THRESHOLDS, **(priors or {}).get("thresholds", {})}["cp"]
    hit = np.nan_to_num(lo, nan=-np.inf) >= th
    out["anomaly"] = hit.astype(int)
    out["severity"] = np.where(np.isfinite(lo), 1.0 / (1.0 + np.exp(-np.nan_to_num(lo))), 0.0)
    w = p["window_min"]
    pheno, mag = PHENO_MEAN, 0.0
    for i in np.flatnonzero(hit):
        if i == 0 or not hit[i - 1]:                     # a run's first reading
            old = z[(t > t[i] - 2 * w) & (t <= t[i] - w)]
            recent = z[(t > t[i] - w) & (t <= t[i])]
            old, recent = old[np.isfinite(old)], recent[np.isfinite(recent)]
            if old.size >= 2 and recent.size >= 2:
                d_mean = abs(recent.mean() - old.mean())
                d_std = abs(recent.std() - old.std())
            else:
                d_mean, d_std = abs(z[i]) if np.isfinite(z[i]) else 0.0, 0.0
            pheno, mag = (PHENO_MEAN, d_mean) if d_mean >= d_std else (PHENO_VAR, d_std)
        out["phenotype"][i] = pheno
        out["magnitude"][i] = mag
    return out


# --- registry metadata (auto-discovered; see wearable/registry.py) ---
LABEL = "Bayesian online CPD (Adams & MacKay 2007)"
PHENOTYPES = (
    'BOCPD_Mean_Shift',
    'BOCPD_Variance_Change',
)
