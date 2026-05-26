"""Hidden Markov Model change-point detection.

Two-state Gaussian HMM (calm vs elevated) with a STICKY transition prior so the
model doesn't flip on noise sub-modes. The pasted 3-state full-covariance model
fired ~44% in contiguous blocks with a bimodal magnitude distribution; the
fix is fewer states, diagonal covariance, and a Dirichlet self-transition
pseudocount that strongly favours staying in the current state.

    detect(time, values) -> (change_types[N], change_degrees[N] in [0,1])

If ``hmm_params.json`` is present and ``HPO_ACTIVE != 1``, its values override
the derived defaults (n_components, covariance_type, sticky).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


_DEFAULT_N_COMPONENTS = 2
_DEFAULT_COVARIANCE_TYPE = "diag"
_DEFAULT_STICKY = 10.0          # Dirichlet pseudocount weight on self-transitions
_DEFAULT_MIN_SEG = 30           # post-hoc: ignore state runs shorter than this
_N_ITER = 100
_RANDOM_STATE = 42

_PARAMS_JSON = Path(__file__).parent / "hmm_params.json"


def _load_overrides() -> dict:
    if os.environ.get("HPO_ACTIVE") == "1" or not _PARAMS_JSON.exists():
        return {}
    try:
        return json.loads(_PARAMS_JSON.read_text())
    except Exception as exc:
        print(f"[HMM] WARNING: failed to load {_PARAMS_JSON.name}: {exc}")
        return {}


def _smooth_states(states: np.ndarray, min_seg: int) -> np.ndarray:
    """Merge runs shorter than ``min_seg`` into their longer neighbour.

    The sticky prior alone is overwhelmed on long series (N=8500 swamps any
    reasonable pseudocount); this post-hoc pass enforces minimum regime length
    in the state sequence itself, so the 'change-point' is only counted when
    the new state actually persists.
    """
    if min_seg <= 1 or len(states) == 0:
        return states
    s = states.copy()
    # iterate until stable (short runs can chain)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(s):
            j = i
            while j < len(s) and s[j] == s[i]:
                j += 1
            run_len = j - i
            if run_len < min_seg and (i > 0 or j < len(s)):
                # merge with the longer neighbour (or the only neighbour at the edge)
                left = s[i - 1] if i > 0 else None
                right = s[j] if j < len(s) else None
                target = left if right is None else (right if left is None else
                         (left if i >= (len(s) - j) else right))
                s[i:j] = target
                changed = True
                break
            i = j
    return s


def detect(time, values):
    N = len(values)
    overrides = _load_overrides()
    n_components = int(overrides.get("n_components", _DEFAULT_N_COMPONENTS))
    covariance_type = str(overrides.get("covariance_type", _DEFAULT_COVARIANCE_TYPE))
    sticky = float(overrides.get("sticky", _DEFAULT_STICKY))
    min_seg = int(overrides.get("min_seg", _DEFAULT_MIN_SEG))

    if N < n_components + 1:
        return np.full(N, "normal", dtype=object), np.zeros(N)

    x_scaled = StandardScaler().fit_transform(np.asarray(values).reshape(-1, 1))

    # Sticky transition prior: diagonal pseudocount >> off-diagonal so the
    # posterior favours self-transitions. Note: with long sequences this prior
    # is overwhelmed by the data, so _smooth_states does the real enforcement.
    transmat_prior = np.ones((n_components, n_components)) + sticky * np.eye(n_components)

    model = GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=_N_ITER,
        random_state=_RANDOM_STATE,
        transmat_prior=transmat_prior,
    )
    model.fit(x_scaled)
    hidden_states = model.predict(x_scaled)
    posteriors = model.predict_proba(x_scaled)

    # Post-hoc: enforce minimum regime length (real fixes for noisy flips).
    hidden_states = _smooth_states(hidden_states, min_seg)

    counts = np.bincount(hidden_states, minlength=n_components)
    minority_state = int(np.argmin(counts))
    change_degrees = posteriors[:, minority_state]

    transition_idx = np.where(np.diff(hidden_states) != 0)[0] + 1
    change_types = np.full(N, "normal", dtype=object)
    for cp in transition_idx:
        change_types[cp] = "shift"
        if cp - 1 >= 0 and change_types[cp - 1] == "normal":
            change_types[cp - 1] = "transition"
        if cp + 1 < N and change_types[cp + 1] == "normal":
            change_types[cp + 1] = "transition"
    return change_types, change_degrees
