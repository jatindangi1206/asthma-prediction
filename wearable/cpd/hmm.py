"""Two-state continuous-time HMM on the whitened residual, filtered forward. Causal.

State 0 is the calm regime, state 1 a volatile one (an autonomic-struggle regime). Both
emit z as a Student-t with mean 0 — state 0 at unit scale, state 1 at `state1_var_mult`
times the variance — so state 1 is a pure, SUSTAINED volatility regime. The t tails
(nu and scale fitted from the patient's past, `priors["nu"]`/`tail_scale`) let a single surge stay in
the calm state; isolated spikes are the Kalman detector's job. Sustained mean offsets
are left to BOCPD / Kalman / drift.

Persistence is stated in minutes, not readings: the generator has rates a = 1/dwell_0
and b = 1/dwell_1, and the transition over an elapsed dt is exact in closed form,

    P00 = b/(a+b) + a/(a+b) e^{-(a+b)dt}        P01 = 1 - P00
    P11 = a/(a+b) + b/(a+b) e^{-(a+b)dt}        P10 = 1 - P11

so a long gap relaxes toward the stationary mix [b, a]/(a+b) on its own and a short
one keeps continuity within a chunk. H8 chunk boundaries reset the state mix.

Only the FORWARD filter P(state 1 | z up to t) is used. The previous version fitted
transitions by EM on the whole chunk and decoded with Viterbi and forward–backward, so
a reading's flag depended on later readings (and, across midnight, on the next day).

Flags every reading where the log-odds of the volatile state is at or over the
calibrated threshold.
"""

import numpy as np
from scipy.special import expit
from wearable.cpd._contracts import series, positive, hit, excess

DEFAULTS = {
    "dwell_normal_min": 1440.0,   # mean stay in the calm state
    "dwell_volatile_min": 120.0,  # mean stay in the volatile state
    "state1_var_mult": 9.0,       # volatile variance = (3 sd)^2 of the calm one
}
THRESHOLDS = {"volatile": 2.2}    # log-odds; fallback without calibration (P ~ 0.9)
PHENOTYPE = "HMM_State_Transition"


def score(time, values, priors=None, **params):
    p = {**DEFAULTS, **params}
    t, z, chunk = series(time, values, priors)
    positive(**p)
    if p["state1_var_mult"] <= 1:
        raise ValueError("state1_var_mult must exceed 1")
    a, b = 1.0 / p["dwell_normal_min"], 1.0 / p["dwell_volatile_min"]
    s = a + b
    v1 = float(p["state1_var_mult"])
    nu = np.broadcast_to(np.asarray((priors or {}).get("nu", 200.0), float), z.shape)
    scale = np.broadcast_to(np.asarray((priors or {}).get("tail_scale", 1.0), float), z.shape)
    if np.any(~np.isfinite(nu)) or np.any(nu <= 2) or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("HMM requires finite nu > 2 and positive tail scales")
    # log-likelihood ratio of volatile vs calm, both Student-t(nu): scale sqrt(v1) vs 1
    ll = lambda x, df: (-0.5 * np.log(v1) - (df + 1) / 2 * (np.log1p(x * x / (df * v1))
                                                         - np.log1p(x * x / df)))
    pv = np.full(len(z), np.nan)
    p1, prev = a / s, None                   # start from the stationary mix
    for i in range(len(z)):
        if not np.isfinite(z[i]):
            continue
        if prev is not None and chunk[i] != chunk[prev]:
            p1 = a / s
        elif prev is not None:
            e = np.exp(-s * (t[i] - t[prev]))
            p1 = p1 * (a / s + b / s * e) + (1 - p1) * (a / s) * (1 - e)
        lo = np.log(p1) - np.log1p(-p1) + ll(z[i] / scale[i], nu[i])
        p1 = float(np.clip(expit(lo), 1e-15, 1 - 1e-15))
        pv[i] = p1
        prev = i
    return {"volatile": _logit(pv)}


def _logit(p):
    # The statistic is calibrated on the log-odds scale: probabilities pile up at
    # 0.9999..., where no threshold can tell two alarms apart.
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p) - np.log1p(-p)


def detect(time, values, priors=None, **params):
    n = len(values)
    out = {"anomaly": np.zeros(n, int),
           "phenotype": np.full(n, "normal", dtype=object),
           "magnitude": np.zeros(n), "severity": np.zeros(n)}
    if n < 2:
        return out
    lo = score(time, values, priors, **params)["volatile"]
    out["scores"] = {"volatile": lo}
    pv = np.where(np.isfinite(lo), 1.0 / (1.0 + np.exp(-np.nan_to_num(lo))), 0.0)
    th = {**THRESHOLDS, **(priors or {}).get("thresholds", {})}["volatile"]
    out["available"] = np.isfinite(lo) & np.isfinite(th)
    fired = hit(lo, th)
    out["anomaly"] = fired.astype(int)
    out["magnitude"] = pv
    out["severity"] = excess(lo, th)
    out["phenotype"] = np.where(fired, PHENOTYPE, "normal").astype(object)
    return out


# --- registry metadata (auto-discovered; see wearable/registry.py) ---
LABEL = "Two-state Student-t forward HMM"
PHENOTYPES = (
    'HMM_State_Transition',
)
