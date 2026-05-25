"""Stage 4 detector — Bayesian Online Change-Point Detection (Adams & MacKay, 2007).

Gaussian observations with a Normal-Gamma conjugate prior (Student-t predictive)
and a constant hazard. Runs on smoothed HRV only.

    detect(time, values) -> (change_types[N], change_degrees[N] in [0,1])

``change_types`` are string labels ('normal'/'shift') to match the other three
detectors; cpd_pipeline converts them to a 0/1 indicator. ``change_degrees`` is
the run-length-0 posterior P(changepoint at t); NaN positions are returned 0.

(Note: this file previously held a chunk-smoothed helper; that step is redundant
because Stage 3 already leaves gaps as NaN. Replaced with real BOCPD per spec.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import gammaln

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline import config


def _student_logpdf(x, mu, var, nu):
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2)
            - 0.5 * np.log(nu * np.pi * var)
            - (nu + 1) / 2 * np.log1p((x - mu) ** 2 / (nu * var)))


def _bocpd(x: np.ndarray, hazard: float) -> np.ndarray:
    """Return P(run length == 0) at each step = changepoint probability."""
    x = (x - x.mean()) / (x.std() or 1.0)   # standardize -> prior scale-invariant
    mu0, kappa0, alpha0, beta0 = 0.0, 1.0, 1.0, 1.0
    mu = np.array([mu0]); kappa = np.array([kappa0])
    alpha = np.array([alpha0]); beta = np.array([beta0])

    n = len(x)
    R = np.zeros(n + 1)
    R[0] = 1.0
    cp = np.zeros(n)
    for t in range(n):
        xt = x[t]
        var = beta * (kappa + 1) / (alpha * kappa)
        nu = 2 * alpha
        pred = np.exp(_student_logpdf(xt, mu, var, nu))

        prev = R[:t + 1]
        new_R = np.empty(t + 2)
        new_R[1:] = prev * pred * (1 - hazard)      # growth
        new_R[0] = np.sum(prev * pred * hazard)     # changepoint mass
        total = new_R.sum()
        if total > 0:
            new_R /= total
        R[:t + 2] = new_R
        cp[t] = new_R[0]

        # conjugate update, prior prepended for the new run-length 0 hypothesis
        new_mu = (kappa * mu + xt) / (kappa + 1)
        new_beta = beta + (kappa * (xt - mu) ** 2) / (2 * (kappa + 1))
        mu = np.concatenate(([mu0], new_mu))
        kappa = np.concatenate(([kappa0], kappa + 1))
        alpha = np.concatenate(([alpha0], alpha + 0.5))
        beta = np.concatenate(([beta0], new_beta))
    return cp


def detect(time, values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    types = np.full(n, "normal", dtype=object)
    degs = np.zeros(n, dtype=float)
    idx = np.where(np.isfinite(values))[0]
    if len(idx) < 5:
        return types, degs
    cp = _bocpd(values[idx], config.BOCPD_HAZARD)
    degs[idx] = np.clip(cp, 0.0, 1.0)
    types[idx[cp > config.BOCPD_PROB_THRESHOLD]] = "shift"
    return types, degs
