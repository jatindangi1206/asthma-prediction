"""Kernel Change Point Detection on the RESIDUAL — online two-window MMD.

Replaces the old offline `ruptures` call with a causal, online detector:

  * window w = 18 (3 h per side): compare the PAST 3h window to the PRESENT 3h.
  * kernel bandwidth σ = median heuristic on the current 2w block (adapts).
  * statistic = Maximum Mean Discrepancy (MMD²) between the two windows.

RECALIBRATION (why this differs from a plain permutation test)
--------------------------------------------------------------
A naive "fire if permutation p < 0.05" over-fires badly on HRV residuals for two
reasons, both fixed here:

1. Circadian heteroscedasticity leaks into the residual — its volatility is
   higher at night than midday — so a raw MMD test flags normal day↔night
   variance transitions as "distribution shifts". FIX: STUDENTIZE the residual by
   a causal slow rolling std before the test, so KCPD only sees departures
   *beyond* the patient's normal time-varying volatility envelope.

2. Statistical significance ≠ clinical relevance: with 36-point windows even tiny
   differences are "significant", and testing one window per step is a massive
   multiple-comparisons problem. FIX: fire on an EFFECT-SIZE gate — the MMD must
   sit MIN_EFFECT_Z null-standard-deviations above its own permutation null
   (a multiplicity-robust criterion that does not depend on the discrete p-floor).

Also: the anomaly is marked at the single event point (the present sample), not
held forward over the stride, so the per-row count equals the number of detected
events (no 3× inflation). The magnitude/severity are still carried forward for
feature continuity.

Fires KCPD_Distribution_Shift; magnitude = the MMD² distance.

NOTE on KCP_Multivariate_Shift: that phenotype is reserved for the multi-channel
extension (a joint MMD over HRV + SpO2 + steps/sleep). On the single residual
channel of the present detection phase it is not applicable, so KCPD emits only
KCPD_Distribution_Shift here; the multivariate variant activates when additional
physiological channels are fused into the residual vector.
"""

import numpy as np

WINDOW = 18                  # 3 hours of 10-min samples per side
N_PERM = 80                  # permutations to build the MMD null (mean/std)
STRIDE = 3                   # evaluate every STRIDE points (magnitude held forward)
MIN_EFFECT_Z = 10.0          # fire only if MMD >= this many null-std above null mean
STUDENTIZE = True            # divide residual by a causal rolling std first
STUD_HALFLIFE = 36           # ~6 h half-life for the rolling-std envelope
PHENOTYPE = "KCPD_Distribution_Shift"
RNG_SEED = 0                 # re-seeded PER detect() call — see below
# NOTE: a module-level RNG made results depend on how many chunks/patients/methods
# had already been processed in the same interpreter (the permutation stream kept
# advancing), so annotating one patient alone disagreed with that same patient
# inside a cohort run. The generator is now created inside detect(), making every
# call order-independent and per-patient reproducible.


def _rbf_gram(x, gamma):
    d2 = (x[:, None] - x[None, :]) ** 2
    return np.exp(-gamma * d2)


def _mmd2_from_K(K, n):
    """Unbiased MMD² for the split [first n | rest n] given full Gram matrix K."""
    Kxx = K[:n, :n]; Kyy = K[n:, n:]; Kxy = K[:n, n:]
    m = n
    sxx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    syy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    sxy = Kxy.sum() / (m * m)
    return sxx + syy - 2.0 * sxy


def _studentize(values, halflife):
    """Causal rolling-std normalization: removes the circadian volatility envelope."""
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    var = None
    s = np.empty_like(values)
    for t in range(values.shape[0]):
        v2 = values[t] ** 2
        var = v2 if var is None else alpha * v2 + (1.0 - alpha) * var
        s[t] = np.sqrt(var)
    s = np.maximum(s, np.median(s[s > 0]) * 1e-3 if np.any(s > 0) else 1e-6)
    return values / s


def detect(time, values, priors=None):
    """Online studentized two-window MMD on the residual. Returns the rich dict."""
    N = len(values)
    out = {"anomaly": np.zeros(N, int),
           "phenotype": np.full(N, "normal", dtype=object),
           "magnitude": np.zeros(N), "severity": np.zeros(N)}
    w = WINDOW
    if N < 2 * w:
        return out

    sig = _studentize(values, STUD_HALFLIFE) if STUDENTIZE else values
    idx2 = np.arange(2 * w)
    rng = np.random.default_rng(RNG_SEED)      # fresh per call -> order-independent

    for t in range(2 * w, N + 1, STRIDE):
        block = sig[t - 2 * w:t]                          # past w | present w
        dists = np.abs(block[:, None] - block[None, :])
        med = np.median(dists[dists > 0]) if np.any(dists > 0) else 1.0
        gamma = 1.0 / (2.0 * med ** 2 + 1e-12)
        K = _rbf_gram(block, gamma)
        obs = _mmd2_from_K(K, w)

        null = np.empty(N_PERM)
        for j in range(N_PERM):
            p = rng.permutation(idx2)
            null[j] = _mmd2_from_K(K[np.ix_(p, p)], w)
        null_mean = null.mean(); null_std = null.std() + 1e-12
        z = (obs - null_mean) / null_std                  # effect size vs the null

        i = t - 1                                          # the present point
        hold = slice(i, min(i + STRIDE, N))               # carry magnitude forward
        out["magnitude"][hold] = max(obs, 0.0)
        out["severity"][hold] = min(max(z / (2.0 * MIN_EFFECT_Z), 0.0), 1.0)
        if z >= MIN_EFFECT_Z:                             # effect-size gate (multiplicity-robust)
            out["anomaly"][i] = 1                          # mark the EVENT point only
            out["phenotype"][i] = PHENOTYPE
    return out
