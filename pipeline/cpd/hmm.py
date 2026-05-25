"""Hidden Markov Model change-point detection.

2-state Gaussian HMM. State transitions identify change points.
No pre-smoothing applied (caller provides VAE-smoothed values).
Extracted from notebooks/hmm.ipynb.
"""

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

import json
from pathlib import Path

_N_COMPONENTS = 3
_COVARIANCE_TYPE = "full"

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

    import os

    # Load optimized parameters if available
    n_components = _N_COMPONENTS
    covariance_type = _COVARIANCE_TYPE

    if os.environ.get("HPO_ACTIVE") != "1":
        config_path = Path(__file__).parent / "hmm_params.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                n_components = int(config.get("n_components", n_components))
                covariance_type = config.get("covariance_type", covariance_type)
                print(f"[HMM] Loaded optimized parameters: n_components={n_components}, covariance={covariance_type}")
            except Exception as e:
                print(f"[HMM] Warning: Failed to load config from {config_path}: {e}")


    if N < n_components + 1:
        return np.full(N, "normal", dtype=object), np.zeros(N)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(values.reshape(-1, 1))

    model = GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=_N_ITER,
        random_state=_RANDOM_STATE,
    )
    model.fit(x_scaled)

    hidden_states = model.predict(x_scaled)
    posteriors = model.predict_proba(x_scaled)


    # Minority state = the state that appears less frequently
    counts = np.bincount(hidden_states, minlength=n_components)
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
