"""Bayesian Online Change Point Detection (Adams & MacKay, 2007) on the RESIDUAL.

Dual-Track Residual Architecture: this runs on residual_hrv = raw − smoothed, so
"normal" is exactly 0. Normal-Normal conjugate model, O(T)-memory rolling form.

Priors (per the architecture spec):
  μ0      = 0.0          (residual normal state)
  hazard  = 1/1000       (≈ one severe exacerbation per ~1000 ten-minute points)
  σ0, σ_x = from the patient's 24-hour residual burn-in (passed in `priors`)
Trigger: changepoint posterior P(run-length → 0) > 0.95  →  BOCPD_Mean_Shift.
"""

import numpy as np

HAZARD_LAMBDA = 1000          # 1/1000 hazard: ~1 event per 1000 ten-minute points
MU0 = 0.0                     # residual baseline is exactly zero
MIN_RUN = 6                   # the MAP run length must have grown >=1h before a collapse counts
DROP_FRAC = 0.5             # collapse = MAP run length falls below this fraction of its prior value
PHENO_MEAN = "BOCPD_Mean_Shift"
PHENO_VAR = "BOCPD_Variance_Change"


def _bocpd(values, hazard_lambda, mu0, sigma0, sigma_x):
    """Rolling BOCPD; returns (cp_prob, map_runlength) per t.

    cp_prob   = P(run length = 0) posterior.
    map_run   = the most-probable run length (its argmax). At a true changepoint
                the posterior mass collapses from a long run onto 0, so map_run
                drops sharply — a hazard-robust changepoint signal (whereas
                P(r=0) never approaches 0.95 under a 1/1000 hazard).
    """
    T = len(values)
    hazard = 1.0 / hazard_lambda
    prior_prec = 1.0 / sigma0 ** 2
    obs_prec = 1.0 / sigma_x ** 2

    rl = np.array([1.0]); mu_s = np.array([mu0]); prec_s = np.array([prior_prec])
    cp = np.empty(T); map_run = np.empty(T)

    for t in range(T):
        x = values[t]; n = len(rl)
        pred_var = 1.0 / prec_s + sigma_x ** 2
        pred = np.exp(-0.5 * (x - mu_s) ** 2 / pred_var) / np.sqrt(2.0 * np.pi * pred_var)
        weighted = pred * rl

        new_rl = np.empty(n + 1)
        new_rl[0] = hazard * weighted.sum()          # reset to run length 0
        new_rl[1:] = (1.0 - hazard) * weighted        # grow run length
        s = new_rl.sum()
        if s > 0.0:
            new_rl /= s
        cp[t] = new_rl[0]
        map_run[t] = float(np.argmax(new_rl))

        new_prec = np.empty(n + 1); new_prec[0] = prior_prec; new_prec[1:] = prec_s + obs_prec
        new_mu = np.empty(n + 1); new_mu[0] = mu0
        new_mu[1:] = (prec_s * mu_s + x * obs_prec) / new_prec[1:]
        rl, mu_s, prec_s = new_rl, new_mu, new_prec
    return cp, map_run


def detect(time, values, priors=None):
    """Detect mean shifts in the residual via run-length collapse. Returns
       {anomaly (0/1), phenotype, magnitude (run-length drop fraction), severity}.
    """
    N = len(values)
    out = {"anomaly": np.zeros(N, int),
           "phenotype": np.full(N, "normal", dtype=object),
           "magnitude": np.zeros(N), "severity": np.zeros(N)}
    if N < 2:
        return out

    priors = priors or {}
    sigma0 = float(priors.get("sigma0", np.std(values) or 1.0)) or 1.0
    cp, map_run = _bocpd(values, HAZARD_LAMBDA, MU0, sigma0, sigma0)

    w = MIN_RUN
    for t in range(1, N):
        prev = map_run[t - 1]
        if prev >= MIN_RUN and map_run[t] <= DROP_FRAC * prev:    # run-length collapse
            # Classify the change CAUSALLY (only data up to t): compare an "old"
            # trailing window to the "recent" window ending at t. A larger mean
            # shift => BOCPD_Mean_Shift; a larger spread change => BOCPD_Variance_Change.
            old = values[max(0, t - 2 * w):t - w]
            recent = values[max(0, t - w):t + 1]
            if old.size >= 2 and recent.size >= 2:
                d_mean = abs(np.mean(recent) - np.mean(old))
                d_std = abs(np.std(recent) - np.std(old))
            else:                                                 # too early -> default to mean
                d_mean, d_std = abs(values[t] - MU0), 0.0
            out["anomaly"][t] = 1
            if d_mean >= d_std:
                out["phenotype"][t] = PHENO_MEAN
                out["magnitude"][t] = d_mean                      # actual mean shift (residual units)
            else:
                out["phenotype"][t] = PHENO_VAR
                out["magnitude"][t] = d_std                       # actual spread change (residual units)
            out["severity"][t] = min(max(d_mean, d_std) / (3.0 * sigma0), 1.0)
    return out
