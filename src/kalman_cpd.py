"""Kalman filter change-point detection.

Level+trend state-space model with adaptive dt.
Innovation score d = nu^2 / S flags anomalous measurements.
Extracted from notebooks/kalman_filter.ipynb (cells 513d340a, 745ad714).
"""

import numpy as np


_H = np.array([[1.0, 0.0]])
_R = np.array([[10.0]])
_Q_FACTOR = 0.1
_THRESHOLD = 60.0  # Lower = more adverse flags


def _transition_and_noise(dt: float) -> tuple[np.ndarray, np.ndarray]:
    A = np.array([[1.0, dt], [0.0, 1.0]])
    Q = _Q_FACTOR * np.array(
        [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]]
    )
    return A, Q


def detect(
    time: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (change_types, change_degrees), both shape (N,).

    change_type: 'normal' | 'medium' | 'adverse'
    change_degree: innovation score normalized to [0, 1] (score / threshold)
    """
    N = len(values)
    x_hat = np.array([[values[0]], [0.0]])
    P = np.eye(2) * 10.0

    scores = np.empty(N)

    for i in range(N):
        if i > 0:
            dt = float(time[i] - time[i - 1])
            A, Q = _transition_and_noise(dt)
            x_pred = A @ x_hat
            P_pred = A @ P @ A.T + Q
        else:
            x_pred = x_hat
            P_pred = P

        nu = values[i] - (_H @ x_pred)[0, 0]
        S = (_H @ P_pred @ _H.T + _R)[0, 0]
        scores[i] = (nu**2) / S

        K = P_pred @ _H.T / S
        x_hat = x_pred + K * nu
        P = (np.eye(2) - K @ _H) @ P_pred

    change_types = np.empty(N, dtype=object)
    change_types[scores <= 8.0] = "normal"  # Lower = more medium/adverse flags
    change_types[(scores > 8.0) & (scores <= _THRESHOLD)] = "medium"
    change_types[scores > _THRESHOLD] = "adverse"

    change_degrees = np.minimum(scores / _THRESHOLD, 1.0)

    return change_types, change_degrees
