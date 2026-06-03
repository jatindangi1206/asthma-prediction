"""Hidden Markov Model change-point detection.

2-state Gaussian HMM. State transitions identify change points.
No pre-smoothing applied (caller provides VAE-smoothed values).
Extracted from notebooks/hmm.ipynb.
"""

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

_N_COMPONENTS = 2
_N_ITER = 100
_RANDOM_STATE = 42


def detect(
    time: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (change_types, change_degrees), both shape (N,).

    change_type: 'normal' | 'transition' | 'shift'
    change_degree: posterior probability of being in the minority state [0, 1]
    """
    N = len(values)

    if N < _N_COMPONENTS + 1:
        return np.full(N, "normal", dtype=object), np.zeros(N)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(values.reshape(-1, 1))

    model = GaussianHMM(
        n_components=_N_COMPONENTS,
        covariance_type="full",
        n_iter=_N_ITER,
        random_state=_RANDOM_STATE,
    )
    model.fit(x_scaled)

    hidden_states = model.predict(x_scaled)
    posteriors = model.predict_proba(x_scaled)

    # Minority state = the state that appears less frequently
    counts = np.bincount(hidden_states, minlength=_N_COMPONENTS)
    minority_state = int(np.argmin(counts))
    change_degrees = posteriors[:, minority_state]

    # Label transitions
    transition_idx = np.where(hidden_states[1:] != hidden_states[:-1])[0] + 1
    change_types = np.full(N, "normal", dtype=object)

    for cp in transition_idx:
        change_types[cp] = "shift"
        if cp - 1 >= 0 and change_types[cp - 1] == "normal":
            change_types[cp - 1] = "transition"
        if cp + 1 < N and change_types[cp + 1] == "normal":
            change_types[cp + 1] = "transition"

    return change_types, change_degrees
