"""Kalman filter change-point detection.

Level+trend state-space model with adaptive dt. Innovation score d = nu^2 / S
flags anomalous measurements.

Two fixes vs the pasted version:
  * **Warm-up burn-in**: detections are suppressed for the first ~5% of points
    so the filter's initial settling (large early innovations as the level
    estimate stabilises) isn't read as a flurry of change points.
  * **Higher observation noise R / lower process noise Q**: trusts the
    observations less, so innovations get divided by a larger S and small
    wobble doesn't trip the threshold.

The decision threshold itself is **derived from this patient's own** post-burn-in
score distribution (99th percentile) — per data characteristic C18 (patient-
specific baselines). If ``kalman_params.json`` is present and ``HPO_ACTIVE != 1``,
its values override the derived defaults.

    detect(time, values) -> (change_types[N], change_degrees[N] in [0,1])
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


_H = np.array([[1.0, 0.0]])
_DEFAULT_R = 100.0              # was 10 — raised to trust observations less
_DEFAULT_Q_FACTOR = 0.01        # was 0.1 — lowered so level moves slowly
_BURN_IN_FRAC = 0.05            # suppress flags in first 5% of points (filter warm-up)
_BURN_IN_MIN = 50

_PARAMS_JSON = Path(__file__).parent / "kalman_params.json"


def _transition_and_noise(dt: float, q_factor: float) -> tuple[np.ndarray, np.ndarray]:
    A = np.array([[1.0, dt], [0.0, 1.0]])
    Q = q_factor * np.array(
        [[dt ** 4 / 4.0, dt ** 3 / 2.0], [dt ** 3 / 2.0, dt ** 2]]
    )
    return A, Q


def _load_overrides() -> dict:
    if os.environ.get("HPO_ACTIVE") == "1" or not _PARAMS_JSON.exists():
        return {}
    try:
        return json.loads(_PARAMS_JSON.read_text())
    except Exception as exc:
        print(f"[Kalman] WARNING: failed to load {_PARAMS_JSON.name}: {exc}")
        return {}


def detect(time, values):
    N = len(values)
    overrides = _load_overrides()
    R = np.array([[float(overrides.get("R", _DEFAULT_R))]])
    q_factor = float(overrides.get("q_factor", _DEFAULT_Q_FACTOR))
    burn_in = int(overrides.get("burn_in", max(_BURN_IN_MIN, int(N * _BURN_IN_FRAC))))

    x_hat = np.array([[values[0]], [0.0]])
    P = np.eye(2) * 10.0
    scores = np.empty(N)

    for i in range(N):
        if i > 0:
            dt = float(time[i] - time[i - 1])
            A, Q = _transition_and_noise(dt, q_factor)
            x_pred = A @ x_hat
            P_pred = A @ P @ A.T + Q
        else:
            x_pred = x_hat
            P_pred = P

        nu = values[i] - (_H @ x_pred)[0, 0]
        S = (_H @ P_pred @ _H.T + R)[0, 0]
        scores[i] = (nu ** 2) / S

        K = P_pred @ _H.T / S
        x_hat = x_pred + K * nu
        P = (np.eye(2) - K @ _H) @ P_pred

    # --- per-patient thresholds derived from post-burn-in score distribution --
    post = scores[burn_in:]
    if len(post) >= 20:
        derived_threshold = float(np.percentile(post, 99))
        derived_normal_th = float(np.percentile(post, 95))
    else:
        derived_threshold, derived_normal_th = 10.0, 2.5

    threshold = float(overrides.get("threshold", derived_threshold))
    normal_th = float(overrides.get("normal_th", derived_normal_th))
    threshold = max(threshold, normal_th)

    change_types = np.empty(N, dtype=object)
    change_types[scores <= normal_th] = "normal"
    change_types[(scores > normal_th) & (scores <= threshold)] = "medium"
    change_types[scores > threshold] = "adverse"
    change_types[:burn_in] = "normal"   # suppress warm-up

    change_degrees = np.minimum(scores / max(threshold, 1e-6), 1.0)
    change_degrees[:burn_in] = 0.0
    return change_types, change_degrees
