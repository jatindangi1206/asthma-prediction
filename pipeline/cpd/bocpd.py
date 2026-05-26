"""Stage 4 detector — Bayesian Online Change-Point Detection (Adams & MacKay, 2007).

Gaussian observations with a Normal-Gamma conjugate prior (Student-t predictive)
and a constant hazard. Runs on smoothed HRV only.

    detect(time, values) -> (change_types[N], change_degrees[N] in [0,1])

Per-patient thresholds: ``transition_th`` and ``shift_th`` default to the 75th /
90th percentiles of THIS patient's nonzero changepoint posteriors (the "knee"
of the magnitude distribution). The hazard prior expects a long run length, so
the posterior only spikes on real regime shifts.

If ``pipeline/cpd/bocpd_params.json`` exists and ``HPO_ACTIVE != 1``, its values
override the derived defaults — keeping HPO as the wiring point.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.special import gammaln

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline import config

_PARAMS_JSON = Path(__file__).parent / "bocpd_params.json"


def _student_logpdf(x, mu, var, nu):
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2)
            - 0.5 * np.log(nu * np.pi * var)
            - (nu + 1) / 2 * np.log1p((x - mu) ** 2 / (nu * var)))


def _bocpd(x: np.ndarray, hazard: float) -> np.ndarray:
    x = (x - x.mean()) / (x.std() or 1.0)
    mu0, kappa0, alpha0, beta0 = 0.0, 1.0, 1.0, 1.0
    mu = np.array([mu0]); kappa = np.array([kappa0])
    alpha = np.array([alpha0]); beta = np.array([beta0])

    n = len(x)
    R = np.zeros(n + 1); R[0] = 1.0
    cp = np.zeros(n)
    for t in range(n):
        xt = x[t]
        var = beta * (kappa + 1) / (alpha * kappa)
        nu = 2 * alpha
        pred = np.exp(_student_logpdf(xt, mu, var, nu))
        prev = R[:t + 1]
        new_R = np.empty(t + 2)
        new_R[1:] = prev * pred * (1 - hazard)
        new_R[0] = np.sum(prev * pred * hazard)
        total = new_R.sum()
        if total > 0:
            new_R /= total
        R[:t + 2] = new_R
        cp[t] = new_R[0]
        new_mu = (kappa * mu + xt) / (kappa + 1)
        new_beta = beta + (kappa * (xt - mu) ** 2) / (2 * (kappa + 1))
        mu = np.concatenate(([mu0], new_mu))
        kappa = np.concatenate(([kappa0], kappa + 1))
        alpha = np.concatenate(([alpha0], alpha + 0.5))
        beta = np.concatenate(([beta0], new_beta))
    return cp


def _load_overrides() -> dict:
    """Read bocpd_params.json (HPO output) when HPO is NOT actively running."""
    if os.environ.get("HPO_ACTIVE") == "1" or not _PARAMS_JSON.exists():
        return {}
    try:
        return json.loads(_PARAMS_JSON.read_text())
    except Exception as exc:
        print(f"[BOCPD] WARNING: failed to load {_PARAMS_JSON.name}: {exc}")
        return {}


def detect(time, values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    types = np.full(n, "normal", dtype=object)
    degs = np.zeros(n, dtype=float)
    idx = np.where(np.isfinite(values))[0]
    if len(idx) < 5:
        return types, degs

    overrides = _load_overrides()
    # Hazard chosen so cp posterior can actually spike above the prior floor on
    # outliers. lambda=100 -> baseline cp ~= 0.01; meaningful spikes above 0.05.
    # Too low and the posterior is dominated by the prior and never moves.
    hazard_lambda = float(overrides.get("hazard_lambda", 100.0))
    hazard = 1.0 / max(hazard_lambda, 1.0)

    cp = _bocpd(values[idx], hazard)
    degs[idx] = np.clip(cp, 0.0, 1.0)

    # Per-patient threshold: 90th percentile of cp values ABOVE the noise floor
    # (5 x hazard). The noise floor strips out the baseline = hazard pileup so
    # the percentile lands on the actual spike distribution, not the prior mass.
    noise_floor = 5.0 * hazard
    spikes = cp[cp > noise_floor]
    if len(spikes) >= 5:
        derived_threshold = float(np.percentile(spikes, 90))
    else:
        derived_threshold = 0.05   # fallback if no spikes detected

    threshold = float(overrides.get("shift_th",
                       overrides.get("threshold", derived_threshold)))
    types[idx[cp > threshold]] = "shift"
    return types, degs
