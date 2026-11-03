"""Kernel Change Point Detection on the whitened residual — online two-window MMD.

Every `stride_min` minutes, compare the PAST window (the `half_min` minutes before the
present window) with the PRESENT window (the last `half_min` minutes) by the unbiased
MMD² under an RBF kernel (bandwidth by the median heuristic on the block). Windows are
defined in clock minutes, so HR and HRV compare the same stretch of time.

Validity rules, from what this cohort's data look like:
  * a window never crosses an H8 chunk reset (the smoother restarted there, so a splice
    would look like a shift) — `priors["chunk"]`;
  * each half must hold at least `coverage` of its expected readings — HRV is irregular
    (one step in ten is longer than 30 min), so strict contiguity would leave few
    windows, while a half that is mostly empty would compare almost nothing;
  * HR is first binned to `bin_min`-minute medians: its readings are correlated over
    ~9 minutes, so 5-minute bins lose little information and shrink the Gram matrices.

The permutation null is valid because the input is the whitened residual, which is
close to independent under no change; on the raw residual (lag-1 correlation 0.78 for
HR) permutation is far too narrow and fired constantly. The statistic is the MMD's
distance above its permutation null in null standard deviations; its threshold is
calibrated on no-change noise from the patient's own data.

The score is set at the present point of each evaluation and held until the next one,
so a shift that stays above threshold across evaluations is one event. Causal: an
evaluation at time T uses readings up to T.
"""

import numpy as np

DEFAULTS = {
    "half_min": 180.0,         # each window's length (3 h)
    "stride_min": 30.0,        # evaluate every 30 min
    "coverage": 0.7,           # share of a half's expected readings it must hold
    "bin_min": 0.0,            # >0: median-bin the readings first (HR: 5)
    "n_perm": 80,              # permutations for the null mean/sd
}
THRESHOLDS = {"shift": 10.0}   # fallback without calibration
PHENOTYPE = "KCPD_Distribution_Shift"
RNG_SEED = 0                   # re-seeded per call: order-independent, reproducible


def _mmd2(K, n):
    """Unbiased MMD² for the split [first n | rest m] of a Gram matrix K."""
    m = K.shape[0] - n
    Kxx, Kyy, Kxy = K[:n, :n], K[n:, n:], K[:n, n:]
    return ((Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
            + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) - 2.0 * Kxy.mean())


def _bins(t, z, chunk, bin_min):
    """-> (time, value, chunk, index of the last reading in the bin), observed only."""
    ok = np.flatnonzero(np.isfinite(z))
    if bin_min <= 0:
        return t[ok], z[ok], chunk[ok], ok
    key = np.floor(t[ok] / bin_min)
    edges = np.flatnonzero((np.diff(key) != 0) | (np.diff(chunk[ok]) != 0)) + 1
    groups = np.split(np.arange(ok.size), edges)
    last = np.array([ok[g[-1]] for g in groups])
    return (t[last], np.array([np.median(z[ok[g]]) for g in groups]), chunk[last], last)


def score(time, values, priors=None, **params):
    p = {**DEFAULTS, **params}
    pr = priors or {}
    t, z = np.asarray(time, float), np.asarray(values, float)
    n = len(z)
    chunk = np.asarray(pr.get("chunk", np.zeros(n, int)))
    step = float(pr.get("step_min") or (np.median(np.diff(t)) if n > 1 else 1.0))
    eff = max(step, p["bin_min"])
    need = max(p["coverage"] * p["half_min"] / eff, 3)
    w = p["half_min"]
    bt, bz, bc, bidx = _bins(t, z, chunk, p["bin_min"])
    out = np.full(n, np.nan)
    if bt.size < 6:
        return {"shift": out}
    # first time of each chunk, to keep both halves inside one chunk
    chunk_start = {c: bt[bc == c][0] for c in np.unique(bc)}
    rng = np.random.default_rng(RNG_SEED)
    grid = np.arange(bt[0] + 2 * w, bt[-1] + 1e-9, p["stride_min"])
    last_i, last_val = None, np.nan
    for T in grid:
        j = np.searchsorted(bt, T, side="right") - 1       # the present point
        if j < 0:
            continue
        c = bc[j]
        if chunk_start[c] > T - 2 * w:
            val = np.nan
        else:
            past = (bt > T - 2 * w) & (bt <= T - w)
            pres = (bt > T - w) & (bt <= T)
            nx, ny = int(past.sum()), int(pres.sum())
            if nx < need or ny < need:
                val = np.nan
            else:
                block = np.r_[bz[past], bz[pres]]
                d = np.abs(block[:, None] - block[None, :])
                med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
                K = np.exp(-d ** 2 / (2.0 * med ** 2 + 1e-12))
                obs = _mmd2(K, nx)
                idx = np.arange(nx + ny)
                null = np.empty(int(p["n_perm"]))
                for r in range(null.size):
                    q = rng.permutation(idx)
                    null[r] = _mmd2(K[np.ix_(q, q)], nx)
                val = (obs - null.mean()) / (null.std() + 1e-12)
        i = int(bidx[j])
        if last_i is not None and i > last_i:
            out[last_i:i] = last_val                       # hold until this evaluation
        last_i, last_val = i, val
    if last_i is not None:
        out[last_i:] = last_val
    out[~np.isfinite(z)] = np.nan
    return {"shift": out}


def detect(time, values, priors=None, **params):
    n = len(values)
    out = {"anomaly": np.zeros(n, int),
           "phenotype": np.full(n, "normal", dtype=object),
           "magnitude": np.zeros(n), "severity": np.zeros(n)}
    if n < 2:
        return out
    s = np.nan_to_num(score(time, values, priors, **params)["shift"], nan=0.0)
    th = {**THRESHOLDS, **(priors or {}).get("thresholds", {})}["shift"]
    hit = s >= th
    out["anomaly"] = hit.astype(int)
    out["phenotype"] = np.where(hit, PHENOTYPE, "normal").astype(object)
    out["magnitude"] = np.maximum(s, 0.0)
    out["severity"] = np.clip(s / (2.0 * th), 0.0, 1.0)
    return out


# --- registry metadata (auto-discovered; see wearable/registry.py) ---
LABEL = "Studentized two-window MMD"
PHENOTYPES = (
    'KCPD_Distribution_Shift',
)
