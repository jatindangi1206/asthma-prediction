"""Kernel Change Point Detection using ruptures.

RBF kernel (best of rbf/linear/cosine from notebook experiments).
Extracted from notebooks/02_kcp_mmd.ipynb.
"""

import numpy as np
import ruptures as rpt

import json
from pathlib import Path

_KERNEL = "rbf"
_MIN_SIZE = 5  # Detect shorter, finer segments
_PENALTY = 12  # Lower penalty = more change points



def detect(
    time: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (change_types, change_degrees), both shape (N,).

    change_type: 'normal' | 'transition' | 'shift'
    change_degree: normalized mean-shift ratio at boundaries, 0.0 elsewhere
    """
    N = len(values)

    import os

    # Load optimized parameters if available
    min_size = _MIN_SIZE
    penalty = _PENALTY

    if os.environ.get("HPO_ACTIVE") != "1":
        config_path = Path(__file__).parent / "kcp_params.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                min_size = int(config.get("min_size", min_size))
                penalty = float(config.get("penalty", penalty))
                print(f"[KCP] Loaded optimized parameters: min_size={min_size}, penalty={penalty:.2f}")
            except Exception as e:
                print(f"[KCP] Warning: Failed to load config from {config_path}: {e}")


    if N < min_size * 2:
        return np.full(N, "normal", dtype=object), np.zeros(N)

    signal = values.reshape(-1, 1)
    algo = rpt.KernelCPD(kernel=_KERNEL, min_size=min_size).fit(signal)
    bkps = algo.predict(pen=penalty)
    change_points = [idx for idx in bkps if idx < N]

    # Robust noise scale for normalising mean shifts
    diffs = np.diff(values)
    mad = np.median(np.abs(diffs - np.median(diffs)))
    sigma = max(1.4826 * mad, 1e-6)

    change_types = np.full(N, "normal", dtype=object)
    change_degrees = np.zeros(N)

    window = 10
    for cp in change_points:
        before = values[max(0, cp - window) : cp]
        after = values[cp : min(N, cp + window)]
        if len(before) == 0 or len(after) == 0:
            continue
        delta = abs(np.mean(after) - np.mean(before))
        ratio = delta / sigma

        degree = min(ratio / 3.0, 1.0)
        change_degrees[cp] = degree

        if ratio >= 0.8:
            change_types[cp] = "shift"
        elif ratio >= 0.3:
            change_types[cp] = "transition"
        # else remains 'normal'

    return change_types, change_degrees

