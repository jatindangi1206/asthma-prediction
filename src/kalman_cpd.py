"""Kalman innovation tracking on the RESIDUAL — drift (CUSUM) + volatility spikes.

Because Track A (the smoother) already removed the baseline, this Kalman filter
in Track B tracks the SLOPE of the residual (expected 0) and watches its
one-step prediction errors (innovations).

Priors (per the architecture spec):
  state         = [level, slope]; expected slope = 0 (flat residual).
  R (meas noise) HIGH  — distrust hardware jitter; seeded from the 24h burn-in std.
  Q (proc noise) LOW   — physiology drifts slowly.
  CUSUM threshold h ≈ 3 — accumulate standardized innovations; a sustained
                          one-directional drift (e.g. HRV creeping down for hours
                          without a sudden BOCPD drop) breaches h.

Phenotypes (all three innovation-based change types in the write-up):
  Kalman_Volatility_Spike       — a single large innovation |z| > h. Magnitude = |z|.
  Kalman_Gradual_Drift          — the CUSUM of innovations breaches h (slow, one-
                                  directional creep). Magnitude = the CUSUM value.
  Kalman_Autocorrelation_Change — the innovation sequence loses whiteness: the
                                  rolling lag-1 autocorrelation of standardized
                                  innovations breaches ACF_H (model/dynamics shift).
                                  Magnitude = |lag-1 autocorrelation|.
"""

import numpy as np

_H = np.array([[1.0, 0.0]])
R_SCALE = 1.0                 # R = (R_SCALE * burn-in std)^2 ; high vs the LOW Q below,
                             #   so innovations are standardized ~N(0,1) under normal data
Q_FACTOR = 1e-3              # "low" process noise
CUSUM_K = 0.5               # CUSUM slack (in z units)
CUSUM_H = 3.0              # CUSUM / spike threshold (≈3 std of baseline noise)
ACF_WIN = 12                # rolling window for the lag-1 innovation autocorrelation (~2h)
ACF_H = 0.6                 # autocorrelation breach threshold (white innovations ~ 0)
PHENO_SPIKE = "Kalman_Volatility_Spike"
PHENO_DRIFT = "Kalman_Gradual_Drift"
PHENO_ACF = "Kalman_Autocorrelation_Change"


def _transition_and_noise(dt):
    A = np.array([[1.0, dt], [0.0, 1.0]])
    Q = Q_FACTOR * np.array([[dt ** 4 / 4.0, dt ** 3 / 2.0], [dt ** 3 / 2.0, dt ** 2]])
    return A, Q


def detect(time, values, priors=None):
    """Track residual slope; flag volatility spikes and CUSUM trend breaks."""
    N = len(values)
    out = {"anomaly": np.zeros(N, int),
           "phenotype": np.full(N, "normal", dtype=object),
           "magnitude": np.zeros(N), "severity": np.zeros(N)}
    if N < 2:
        return out

    priors = priors or {}
    sigma0 = float(priors.get("sigma0", np.std(values) or 1.0)) or 1.0
    R = np.array([[(R_SCALE * sigma0) ** 2]])             # HIGH measurement noise

    x_hat = np.array([[values[0]], [0.0]])               # level = first residual, slope = 0
    P = np.eye(2) * (sigma0 ** 2)
    cusum_pos = 0.0; cusum_neg = 0.0                     # two-sided CUSUM of z
    zbuf = []                                            # recent standardized innovations (for ACF)

    for i in range(N):
        if i > 0:
            dt = float(time[i] - time[i - 1]) or 1.0
            A, Q = _transition_and_noise(dt)
            x_pred = A @ x_hat
            P_pred = A @ P @ A.T + Q
        else:
            x_pred, P_pred = x_hat, P

        nu = values[i] - (_H @ x_pred)[0, 0]
        S = (_H @ P_pred @ _H.T + R)[0, 0]
        z = nu / np.sqrt(S)                              # standardized innovation
        out["magnitude"][i] = abs(z)
        out["severity"][i] = min(abs(z) / (2.0 * CUSUM_H), 1.0)

        # two-sided CUSUM of standardized innovations
        cusum_pos = max(0.0, cusum_pos + z - CUSUM_K)
        cusum_neg = max(0.0, cusum_neg - z - CUSUM_K)

        # rolling lag-1 autocorrelation of the (causal) innovation window
        zbuf.append(z)
        if len(zbuf) > ACF_WIN:
            zbuf.pop(0)
        acf1 = 0.0
        if len(zbuf) == ACF_WIN:
            za = np.asarray(zbuf); za = za - za.mean()
            denom = float(np.sum(za * za))
            acf1 = float(np.sum(za[1:] * za[:-1]) / denom) if denom > 1e-9 else 0.0

        if abs(z) > CUSUM_H:                             # 1) sudden volatility spike
            out["anomaly"][i] = 1; out["phenotype"][i] = PHENO_SPIKE
            out["magnitude"][i] = abs(z)                 # peak Z-score
        elif abs(acf1) > ACF_H:                          # 2) autocorrelation (whiteness) change
            out["anomaly"][i] = 1; out["phenotype"][i] = PHENO_ACF
            out["magnitude"][i] = abs(acf1)              # lag-1 autocorrelation
            out["severity"][i] = min(abs(acf1), 1.0)
        elif cusum_pos > CUSUM_H or cusum_neg > CUSUM_H:  # 3) slow one-directional drift
            cval = max(cusum_pos, cusum_neg)
            out["anomaly"][i] = 1; out["phenotype"][i] = PHENO_DRIFT
            out["magnitude"][i] = cval                   # cumulative sum of the creep
            out["severity"][i] = min(cval / (2.0 * CUSUM_H), 1.0)
            cusum_pos = 0.0; cusum_neg = 0.0            # reset after flagging the drift

        K = P_pred @ _H.T / S
        x_hat = x_pred + K * nu
        P = (np.eye(2) - K @ _H) @ P_pred
    return out
