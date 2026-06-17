"""Hidden Markov Model regime detection on the RESIDUAL.

Detects when the patient toggles from a Homeostasis regime (State 0) into an
Exacerbation regime (State 1), per the architecture spec:

  * exactly 2 states.
  * sticky transition matrix: diagonal 0.95, off-diagonal 0.05 (biology is sticky
    — a mathematical shock-absorber against false alarms).
  * emissions: both states have mean 0 (we work on the residual); State 0 has LOW
    variance (calm autonomic tone), State 1 has HIGH variance (autonomic
    struggle). Variances are seeded from the 24h burn-in std (`priors`).

State 1 decoded by Viterbi  →  HMM_State_Transition; magnitude = P(State 1).
The means are held fixed at 0, so State 1 is a pure VOLATILITY regime (sustained
mean offsets are left to BOCPD / Kalman).
"""

import logging
import numpy as np
from hmmlearn.hmm import GaussianHMM

logging.getLogger("hmmlearn").setLevel(logging.ERROR)   # silence per-chunk convergence logs

STICKINESS = 0.95            # P(stay in current state)
STATE1_VAR_MULT = 9.0       # State-1 variance = (3*std)^2 = 9 * burn-in variance
PHENOTYPE = "HMM_State_Transition"
_MIN_ROWS = 12


def detect(time, values, priors=None):
    """2-state sticky HMM on the residual; flag the high-variance (State 1) regime."""
    N = len(values)
    out = {"anomaly": np.zeros(N, int),
           "phenotype": np.full(N, "normal", dtype=object),
           "magnitude": np.zeros(N), "severity": np.zeros(N)}
    if N < _MIN_ROWS:
        return out

    priors = priors or {}
    var0 = float(priors.get("sigma0", np.std(values) or 1.0)) ** 2 or 1.0

    try:
        model = GaussianHMM(n_components=2, covariance_type="diag",
                            n_iter=100, random_state=42,
                            init_params="", params="t")      # emissions HARDCODED; EM learns only transitions
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[STICKINESS, 1 - STICKINESS],
                                    [1 - STICKINESS, STICKINESS]])
        model.means_ = np.array([[0.0], [0.0]])              # both states centred at 0
        model.covars_ = np.array([[var0], [var0 * STATE1_VAR_MULT]])  # calm vs volatile

        x = values.reshape(-1, 1)
        model.fit(x)
        states = model.predict(x)                            # Viterbi
        post = model.predict_proba(x)
    except Exception:
        return out

    # State 1 = the higher-variance state (identify after EM in case of relabeling)
    vol_state = int(np.argmax(model.covars_.reshape(-1)))
    p_vol = post[:, vol_state]
    in_vol = (states == vol_state).astype(int)

    out["anomaly"] = in_vol
    out["magnitude"] = p_vol
    out["severity"] = p_vol
    out["phenotype"] = np.where(in_vol == 1, PHENOTYPE, "normal").astype(object)
    return out
