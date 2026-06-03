"""Bayesian Online Change Point Detection (Adams & MacKay, 2007).

Normal-Normal conjugate model. Memory-efficient rolling implementation:
O(T) space instead of the O(T^2) R-matrix in the notebook.
Extracted from notebooks/01_bocpd.ipynb (cell 11).
"""

import numpy as np

_HAZARD_LAMBDA = 150  # Lower = more changepoints
_CP_THRESHOLD_TRANSITION = 0.25  # Lower = more transitions
_CP_THRESHOLD_SHIFT = 0.55  # Lower = more shifts


def _bocpd(values: np.ndarray, hazard_lambda: int, mu0: float, sigma0: float, sigma_x: float) -> np.ndarray:
    """Returns changepoint_prob array of shape (T,)."""
    T = len(values)
    hazard = 1.0 / hazard_lambda
    prior_prec = 1.0 / sigma0**2
    obs_prec = 1.0 / sigma_x**2

    # Rolling run-length distribution and conjugate sufficient stats
    rl = np.array([1.0])
    mu_s = np.array([mu0])
    prec_s = np.array([prior_prec])

    changepoint_prob = np.empty(T)

    for t in range(T):
        x = values[t]
        n = len(rl)

        pred_var = 1.0 / prec_s + sigma_x**2
        pred_probs = (
            np.exp(-0.5 * (x - mu_s) ** 2 / pred_var)
            / np.sqrt(2.0 * np.pi * pred_var)
        )

        weighted = pred_probs * rl
        new_rl = np.empty(n + 1)
        new_rl[0] = hazard * weighted.sum()          # change: reset to 0
        new_rl[1:] = (1.0 - hazard) * weighted      # growth: r → r+1

        total = new_rl.sum()
        if total > 0.0:
            new_rl /= total

        changepoint_prob[t] = new_rl[0]

        # Update sufficient stats for grown runs (index 0 stays at prior)
        new_prec = np.empty(n + 1)
        new_prec[0] = prior_prec
        new_prec[1:] = prec_s + obs_prec

        new_mu = np.empty(n + 1)
        new_mu[0] = mu0
        new_mu[1:] = (prec_s * mu_s + x * obs_prec) / new_prec[1:]

        rl = new_rl
        mu_s = new_mu
        prec_s = new_prec

    return changepoint_prob


def detect(
    time: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (change_types, change_degrees), both shape (N,).

    change_type: 'normal' | 'transition' | 'shift'
    change_degree: changepoint posterior probability [0, 1]
    """
    if len(values) < 2:
        return np.full(len(values), "normal", dtype=object), np.zeros(len(values))

    mu0 = float(np.median(values))
    sigma0 = float(np.std(values)) or 1.0
    sigma_x = float(np.median(np.abs(np.diff(values))) / 0.6745) or 1.0

    prob = _bocpd(values, _HAZARD_LAMBDA, mu0, sigma0, sigma_x)

    change_types = np.empty(len(values), dtype=object)
    change_types[prob <= _CP_THRESHOLD_TRANSITION] = "normal"
    change_types[
        (prob > _CP_THRESHOLD_TRANSITION) & (prob <= _CP_THRESHOLD_SHIFT)
    ] = "transition"
    change_types[prob > _CP_THRESHOLD_SHIFT] = "shift"

    return change_types, prob
