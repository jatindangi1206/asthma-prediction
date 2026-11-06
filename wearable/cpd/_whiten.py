"""The per-patient, per-channel noise model every detector reads through. Causal.

Why it exists: the detectors assume independent noise of a known scale, and the residual
is neither. Measured on this cohort, the HR residual has lag-1 correlation 0.78 at 1 min
and 11% of readings beyond 3 sigma; the HRV residual's spread is ~2x larger by day than
by night. Fed raw, those properties fire the detectors on pure noise. So the residual is
first turned into standardised innovations `z` that are close to independent with unit
scale under no change, whatever the cadence or the gaps:

    u_t = ((r_t - c[h_t]) / s[h_t] - m[k_t]) / q[k_t]
          centre and scale by hour of day, then correct for the behavioural state k
          (sleep / active / rest, from the sleep and steps flags)
    u   = x + e,  x: OU with correlation time tau, variance a;  e: white, variance 1-a
    z_t = (u_t - E[u_t | past]) / sd[u_t | past]  a scalar Kalman filter on that model

After a long gap the OU memory has decayed (exp(-dt/tau) -> 0) and z_t = u_t; after a
short one the filter keeps continuity. At an H8 chunk start the smoother restarted, so
the memory is cleared outright. Nothing resets the detectors themselves.

Everything is fitted on the PAST only: parameters are refitted at arming and at each
later midnight from the trailing `baseline_days` of readings, skipping days that looked
abnormal (more than `abnormal_frac` of |z| > 3), so a real decline cannot become the new
normal. Nothing is pooled across patients: each patient's own first worn day
(`burn_in_min` of readings) starts the baseline, and detectors stay unarmed until then.

Thresholds come from a no-change twin of the patient (`twin`): whole days of the
patient's own trailing data, resampled in shuffled order, whitened by this same noise
model. Each detector's score is run on the twin and the threshold set so that it fires
`rate` times per worn day. At arming the twin is built from the first worn day alone;
every `recal_days` it is rebuilt from the trailing `baseline_days`, minus abnormal days.

HR is first median-binned to `bin_min` minutes (`bins`): its 1–5-minute surges are
posture and movement, not the hours-to-days changes these detectors are for.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from wearable.config import GAP_THRESHOLD_MIN
from wearable.cpd._contracts import positive, series

# Step-level settings (the `detect` block of config.CPD_SETTINGS overrides them per
# channel). Everything about time is in minutes.
DEFAULTS = {
    "step_min": 10.0,          # nominal cadence: worn minutes per reading
    "burn_in_min": 1440.0,     # worn minutes before anything arms (one worn day)
    "baseline_days": 14.0,     # trailing window the noise model is refitted on
    "abnormal_frac": 0.3,      # a day with more than this share of |z| > 3 is not baseline
    "reentry_gap_min": 60.0,   # a gap longer than this is a re-entry
    "settle_min": 20.0,        # after a re-entry: update but do not alarm for this long
    "settle_h8_min": 30.0,     # after an H8 chunk reset (smoother restarted)
    "null_days": 60.0,         # worn days of the no-change twin used for calibration
    "recal_days": 7.0,         # thresholds are recalibrated this often, on trailing days
    "bin_min": 0.0,            # >0: median-bin the readings first (HR: 5)
    "state_window_min": 1.0,   # minutes a reading summarises, for its sleep/activity state
    "max_null_occupancy": 0.1, # candidate per-stream guard, dimensionless; not a clinical target
}
_HOUR_N0 = 20                  # pseudo-count shrinking an hour's centre/scale to the day's
_STATE_N0 = 50                 # pseudo-count shrinking a state's offset/ratio to none
# Behavioural state of a reading (from the ingested sleep_active / steps_active flags):
UNKNOWN, SLEEP, ACTIVE, REST = 0, 1, 2, 3
_Z_ABN = 3.0
_HUBER = 3.0                   # robust-update cutoff, in innovation sd


def _robust(x):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    med = float(np.median(x))
    return med, float(max(1.4826 * np.median(np.abs(x - med)), 1e-6))


def _hourly(r, hour):
    """Centre and scale per hour of day, each shrunk toward the whole window's, then
    smoothed around the clock (±2 h, weights 1-2-3-2-1). One hour holds only ~50
    effective readings of HR over two weeks, so a raw per-hour MAD is off by ~15%; the
    circadian profile itself is smooth, so pooling neighbours costs little and keeps
    the detectors from reading estimation noise as a volatility change."""
    c0, s0 = _robust(r)
    c, s = np.full(24, c0), np.full(24, s0)
    for h in range(24):
        x = r[hour == h]
        n = int(np.isfinite(x).sum())
        if n >= 3:
            ch, sh = _robust(x)
            c[h] = (n * ch + _HOUR_N0 * c0) / (n + _HOUR_N0)
            s[h] = (n * sh + _HOUR_N0 * s0) / (n + _HOUR_N0)
    w = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0
    circ = lambda v: sum(wk * np.roll(v, k - 2) for k, wk in enumerate(w))
    return circ(c), np.exp(circ(np.log(s)))


def _by_state(u, state):
    """Per state (sleep / active / rest): the centre offset and scale ratio of the
    hour-normalised residual u, each shrunk toward none (0, 1) by `_STATE_N0`.

    Measured on this cohort, clock hour explains the residual's spread about twice as
    well as state, but state adds to it (sleep at night and sleep by day differ), so the
    state is a correction on top of the hour profile, not a replacement for it. UNKNOWN
    (no flags) gets no correction."""
    off, ratio = np.zeros(4), np.ones(4)
    _, s_all = _robust(u)
    for k in (SLEEP, ACTIVE, REST):
        x = u[state == k]
        n = int(np.isfinite(x).sum())
        if n >= 10:
            ck, sk = _robust(x)
            off[k] = n * ck / (n + _STATE_N0)
            ratio[k] = (n * sk / s_all + _STATE_N0) / (n + _STATE_N0)
    return off, ratio


def _ou_fit(t, u, chunk, step):
    """(a, tau): u's correlation at lag L is a*exp(-L/tau). Pairs never cross a chunk."""
    rows = []
    for k in (1, 2, 3, 5):
        lag = k * step
        tol = 0.3 * lag
        j = np.searchsorted(t, t + lag - tol)
        j = np.minimum(j, len(t) - 1)
        ok = (np.abs(t[j] - t - lag) <= tol) & (chunk[j] == chunk) & (j > np.arange(len(t)))
        ok &= np.isfinite(u) & np.isfinite(u[j])
        if ok.sum() >= 30:
            rho = float(np.corrcoef(u[ok], u[j[ok]])[0, 1])
            if rho > 0.02:
                rows.append((lag, rho))
    if not rows:
        return 0.0, step
    if len(rows) == 1:
        return float(min(rows[0][1], 0.99)), float(step)
    L, R = np.array(rows).T
    slope, icpt = np.polyfit(L, np.log(R), 1)
    tau = float(np.clip(-1.0 / slope, step / 2, 720.0)) if slope < 0 else 720.0
    return float(np.clip(np.exp(icpt), 0.0, 0.99)), tau


def _filter(idx, t, u, chunk, a, tau, state=(0.0, None, None)):
    """Innovations of u (aligned with the global indices `idx`) under OU(a, tau) plus
    white(1-a), with a Huber-robust update. `state` = (x, P, previous global index)
    may be initialized by replay under the current fit."""
    e = max(1.0 - a, 1e-2)
    x, P, prev = state
    P = a if P is None else P
    z = np.full(len(idx), np.nan)
    for k, i in enumerate(idx):
        if not np.isfinite(u[k]):
            continue
        if prev is None or chunk[i] != chunk[prev]:
            phi = 0.0                                  # H8 reset: memory cleared
        else:
            phi = np.exp(-(t[i] - t[prev]) / tau)
        x, P = phi * x, phi * phi * P + a * (1 - phi * phi)
        S = P + e
        nu = u[k] - x
        z[k] = nu / np.sqrt(S)
        # Huber-weighted update: a reading beyond _HUBER sd moves the state less, so one
        # surge does not drag the state and echo into the next innovations
        K = P / S * min(1.0, _HUBER / max(abs(z[k]), 1e-12))
        x, P = x + K * nu, (1 - K) * P
        prev = i
    return z, (x, P, prev)


def settle_mask(t, chunk, cfg):
    """True where a reading is inside a settle window after a re-entry or chunk reset."""
    out = np.zeros(len(t), bool)
    start, length = None, 0.0
    for i in range(len(t)):
        if i == 0 or chunk[i] != chunk[i - 1]:
            start, length = t[i], cfg["settle_h8_min"]
        elif t[i] - t[i - 1] > cfg["reentry_gap_min"]:
            start, length = t[i], cfg["settle_min"]
        out[i] = t[i] - start < length
    return out


def noise_model(stamp, t, r, raw, chunk, cfg, state=None, exposure=None):
    """Past-only fits; restart burn-in after history ages beyond baseline_days.

    Exposure is observed-support minutes, not the duration of gaps between samples.
    Each epoch has independent calibration and filter state.
    """
    t, r, chunk = series(t, r, {"chunk": chunk})
    stamp, raw = np.asarray(stamp, "datetime64[ns]"), np.asarray(raw, float)
    n = len(r)
    if stamp.shape != r.shape or raw.shape != r.shape:
        raise ValueError("noise-model timestamps/raw must align with the selected series")
    exposure = (np.full(n, cfg["step_min"], float) if exposure is None
                else np.asarray(exposure, float))
    if exposure.shape != r.shape or not np.isfinite(exposure).all() or np.any(exposure <= 0):
        raise ValueError("noise-model exposure must be aligned, finite and positive")
    edges = np.r_[0, np.flatnonzero(np.diff(t) >= cfg["baseline_days"] * 1440) + 1, n]
    result = None
    fits, abnormal, arms = [], set(), []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == lo:
            continue
        out = _noise_epoch(stamp[lo:hi], t[lo:hi], r[lo:hi], raw[lo:hi],
                           chunk[lo:hi], cfg, None if state is None else state[lo:hi],
                           exposure[lo:hi])
        if result is None:
            result = {k: np.full(n, np.nan) for k in
                      ("z", "scale", "reentry_jump_z", "reentry_gap_min", "tail_nu", "tail_scale")}
            result.update(armed=np.zeros(n, bool), settle=np.zeros(n, bool))
        for k in result:
            result[k][lo:hi] = out[k]
        if out["arm"] is not None:
            arms.append(lo + out["arm"])
        fits.extend(out["fits"])
        abnormal.update(out["abnormal"])
    if result is None:
        result = {k: np.empty(0) for k in
                  ("z", "scale", "reentry_jump_z", "reentry_gap_min", "tail_nu", "tail_scale")}
        result.update(armed=np.zeros(0, bool), settle=np.zeros(0, bool))
    initial_tail = result["tail_nu"][np.isfinite(result["tail_nu"])]
    result.update(arm=arms[0] if arms else None, arms=arms, fits=fits,
                  abnormal=abnormal, nu=float(initial_tail[0]) if initial_tail.size else 200.0,
                  exposure=exposure)
    return result


def _noise_epoch(stamp, t, r, raw, chunk, cfg, state, exposure):
    """All inputs are the patient's OBSERVED readings, time-sorted.

    stamp: datetime64[ns] (local clock), t: minutes since the first reading,
    r: the column detectors read, raw: the channel's own value, chunk: H8 chunk ids,
    state: SLEEP / ACTIVE / REST / UNKNOWN per reading (None = all UNKNOWN).
    """
    n = len(r)
    state = np.zeros(n, int) if state is None else np.asarray(state, int)
    step = float(cfg["step_min"])
    hour = ((stamp.astype("datetime64[h]") - stamp.astype("datetime64[D]"))
            .astype(int)).astype(int)
    day = stamp.astype("datetime64[D]")
    worn = np.cumsum(exposure)
    out = {"z": np.full(n, np.nan), "scale": np.full(n, np.nan),
           "armed": np.zeros(n, bool), "settle": settle_mask(t, chunk, cfg),
           "reentry_jump_z": np.full(n, np.nan), "reentry_gap_min": np.full(n, np.nan),
           "arm": None, "fits": [], "abnormal": set(),
           "tail_nu": np.full(n, np.nan), "tail_scale": np.full(n, np.nan)}
    reached = np.flatnonzero(worn >= cfg["burn_in_min"])
    if reached.size == 0 or reached[0] + 1 >= n:
        return out
    arm = int(reached[0]) + 1                         # first reading AFTER the worn day
    out["arm"] = arm
    # refits: at arming, then at the first reading of every later calendar day
    new_day = np.flatnonzero(day[1:] != day[:-1]) + 1
    refits = [arm] + [int(i) for i in new_day if i > arm]
    abnormal = set()
    raw_scale = np.full(n, np.nan)
    for k, a0 in enumerate(refits):
        b0 = refits[k + 1] if k + 1 < len(refits) else n
        lo = np.searchsorted(stamp, stamp[a0] - np.timedelta64(int(cfg["baseline_days"] * 1440), "m"))
        base = np.arange(lo, a0)
        if abnormal:
            base = base[~np.isin(day[base], list(abnormal))]
        if base.size < 10:
            out["fits"].append({"from": stamp[a0], "status": "insufficient_healthy_history"})
            continue
        c, s = _hourly(r[base], hour[base])
        u_h = (r[base] - c[hour[base]]) / s[hour[base]]
        off, ratio = _by_state(u_h, state[base])
        u_base = (u_h - off[state[base]]) / ratio[state[base]]
        a, tau = _ou_fit(t[base], u_base, chunk[base], step)
        # Replay past observations in the NEW coordinates instead of carrying x/P
        # from an incompatible normalization. No future rows enter the replay.
        z_pool, kf = _filter(base, t, u_base, chunk, a, tau)
        pool = z_pool[np.isfinite(z_pool)]
        nu_, tail_scale = 200.0, 1.0
        if pool.size > 30 and np.std(pool) > 1e-8:
            try:
                fitted = student_t.fit(pool, floc=0.0)
                if np.isfinite(fitted).all() and fitted[2] > 1e-8:
                    nu_, tail_scale = float(np.clip(fitted[0], 2.5, 200.0)), float(fitted[2])
            except (ValueError, RuntimeError, FloatingPointError):
                pass   # declared conservative Gaussian-like tail fallback
        seg = np.arange(a0, b0)
        u = ((r[seg] - c[hour[seg]]) / s[hour[seg]] - off[state[seg]]) / ratio[state[seg]]
        z, kf = _filter(seg, t, u, chunk, a, tau, kf)
        out["z"][seg], out["scale"][seg] = z, s[hour[seg]] * ratio[state[seg]]
        out["tail_nu"][seg], out["tail_scale"][seg] = nu_, tail_scale
        out["fits"].append({"from": stamp[a0], "status": "fitted", "n_history": len(base),
                            "hour_center": c.tolist(), "hour_scale": s.tolist(),
                            "state_offset": off.tolist(), "state_ratio": ratio.tolist(),
                            "ou_a": a, "ou_tau": tau, "tail_nu": nu_, "tail_scale": tail_scale})
        raw_scale[seg] = _robust(raw[base])[1]
        for d in np.unique(day[seg]):
            m = day[seg] == d
            zz = z[m][np.isfinite(z[m])]
            if zz.size and np.mean(np.abs(zz) > _Z_ABN) > cfg["abnormal_frac"]:
                abnormal.add(d)
    out["armed"] = (np.arange(n) >= arm) & ~out["settle"] & np.isfinite(out["z"])
    out["abnormal"] = abnormal
    _reentry(t, raw, chunk, raw_scale, out, cfg)
    return out


def _reentry(t, raw, chunk, raw_scale, out, cfg):
    """A level change across a gap is real but undated: record it, never alarm on it.
    Pre = median raw over the settle length before the gap, post = over the settle
    window after it, in units of the baseline's robust spread of raw. Written at the
    first reading after the settle window, when it is first known."""
    arm = out["arm"]
    for i in range(max(arm, 1), len(t)):
        gap = t[i] - t[i - 1]
        h8 = chunk[i] != chunk[i - 1]
        if not (h8 or gap > cfg["reentry_gap_min"]):
            continue
        L = cfg["settle_h8_min"] if h8 else cfg["settle_min"]
        pre = raw[(t > t[i - 1] - L) & (t <= t[i - 1])]
        post_idx = np.flatnonzero((t >= t[i]) & (t < t[i] + L))
        after = np.flatnonzero(t >= t[i] + L)
        if not pre.size or not post_idx.size or not after.size:
            continue
        j = after[0]
        if np.isfinite(raw_scale[j]) and raw_scale[j] > 0:
            out["reentry_jump_z"][j] = (np.median(raw[post_idx]) - np.median(pre)) / raw_scale[j]
            out["reentry_gap_min"][j] = gap


def reading_state(stamp, sleep, steps, window_min):
    """SLEEP / ACTIVE / REST / UNKNOWN for each reading, from minute flag series.

    sleep, steps: pandas Series of 0/1 indexed by minute (the ingested sleep_active /
    steps_active). A reading summarises the `window_min` minutes ending at it (HRV's
    SDNN covers ~10 min; HR is one minute): SLEEP if most of them were sleep, else
    ACTIVE if any had steps, else REST. No flags in the window: UNKNOWN."""
    m = pd.DatetimeIndex(stamp).floor("min")
    w = f"{int(max(window_min, 1))}min"
    one = lambda x: x.groupby(level=0).max().sort_index()      # one value per minute
    sl = one(sleep).rolling(w).mean().reindex(m).to_numpy()
    st = one(steps).rolling(w).max().reindex(m).to_numpy()
    out = np.full(len(m), UNKNOWN)
    awake = np.isfinite(sl) & (sl < 0.5) & np.isfinite(st)
    out[np.isfinite(sl) & (sl >= 0.5)] = SLEEP
    out[awake & (st >= 1)] = ACTIVE
    out[awake & (st < 1)] = REST
    return out


# ---- calibration ----------------------------------------------------------------------
def events(flag):
    """Number of runs of True (rising edges)."""
    f = np.asarray(flag, bool)
    return int(np.sum(f & ~np.r_[False, f[:-1]]))


def bins(stamp, t, r, raw, chunk, bin_min, state=None, reading_min=1.0):
    """Median-bin the readings into `bin_min`-minute bins that never cross a chunk.

    Returns the binned series (stamp and time of each closed bin's boundary, the medians,
    the chunk) and, for every reading, the bin whose output it holds: the latest bin
    whose close is at or before it (-1 before the first bin closes). Open terminal
    bins are omitted. Callers must also enforce chunk, settling and expiry masks.

    For HR this matters more than speed: its readings are correlated over ~9 min, so
    5-minute medians lose little information, and they absorb the 1–5-minute surges
    (posture, movement) that otherwise dominate the residual's tails."""
    positive(bin_min=bin_min, reading_min=reading_min)
    if not len(t):
        return {"stamp": stamp, "t": t, "r": r, "raw": raw, "chunk": chunk,
                "last": np.empty(0, int), "holder": np.empty(0, int),
                "state": np.empty(0, int), "exposure": np.empty(0)}
    key = np.floor(t / bin_min)
    edge = np.r_[True, (key[1:] != key[:-1]) | (chunk[1:] != chunk[:-1])]
    gid = np.cumsum(edge) - 1
    last = np.r_[np.flatnonzero(edge)[1:] - 1, len(t) - 1]
    med = lambda v: pd.Series(v).groupby(gid).median().to_numpy()
    close = (key[last] + 1) * bin_min
    closed = close <= t[-1]             # never publish an open terminal bin
    ends = np.flatnonzero(closed)
    close = close[closed]
    publish = np.searchsorted(t, close, side="left")
    holder = np.searchsorted(close, t, side="right") - 1
    st = np.zeros(len(last), int)
    if state is not None:                         # the bin's majority state
        st = (pd.Series(np.asarray(state, int)).groupby(gid)
              .agg(lambda x: x.value_counts().idxmax()).to_numpy())
    sizes = np.bincount(gid)
    return {"stamp": stamp[0] + (close * 60e9).astype("timedelta64[ns]"),
            "t": close, "r": med(r)[ends], "raw": med(raw)[ends],
            "chunk": chunk[last][ends], "last": publish, "holder": holder,
            "state": st[ends], "exposure": np.minimum(sizes[ends] * reading_min, bin_min)}


def twin(stamp, r, raw, chunk, days, cfg, seed=0, state=None, cutoff=None,
         lower=None, exposure=None):
    """A no-change twin of the patient, whitened by this same noise model.

    Whole days of the patient's own series (`days`: calendar dates to draw from) are
    resampled with replacement and laid end to end on consecutive dates, each keeping
    its own clock times and behavioural states, out to `null_days` worn days (plus a
    burn-in day for the twin's own noise model). Within a day everything is real:
    bursts, tails, the day/night pattern, sleep and activity, the gaps. Across days the order is shuffled, so a trend or a lasting change
    cannot survive unchanged. Internal chunks are preserved and every resampled day
    starts a new block. The twin's z therefore
    carries the same estimation effects as the real z, and a threshold set on it means
    the same thing on the real data. Uses only the days it is given: causal."""
    rng = np.random.default_rng(seed)
    dd = stamp.astype("datetime64[D]")
    valid = np.ones(len(stamp), bool)
    if cutoff is not None:
        valid &= stamp < cutoff
    if lower is not None:
        valid &= stamp >= lower
    pools = [np.flatnonzero((dd == d) & valid) for d in days]
    pools = [i for i in pools if i.size]
    if not pools:
        raise ValueError("no eligible past days for shuffle calibration")
    exposure = np.full(len(stamp), cfg["step_min"]) if exposure is None else exposure
    exposure = np.asarray(exposure, float)
    if exposure.shape != r.shape or not np.isfinite(exposure).all() or np.any(exposure <= 0):
        raise ValueError("shuffle exposure must be aligned, finite and positive")
    positive(null_days=cfg["null_days"], burn_in_min=cfg["burn_in_min"])
    step = float(cfg["step_min"])
    need = cfg["null_days"] * 1440 + cfg["burn_in_min"]
    picks, total = [], 0
    while total < need:
        i = pools[rng.integers(len(pools))]
        picks.append(i)
        total += float(np.sum(exposure[i]))
    start = np.datetime64("2000-01-03")
    st = np.concatenate([start + k * np.timedelta64(1, "D") + (stamp[i] - dd[i])
                         for k, i in enumerate(picks)])
    parts, offset = [], 0
    for i in picks:
        boundary = np.r_[True, (np.diff(chunk[i]) != 0) |
                         (np.diff(stamp[i]).astype("timedelta64[s]").astype(float) / 60 >= GAP_THRESHOLD_MIN)]
        ids = np.cumsum(boundary) + offset
        parts.append(ids)
        offset = int(ids[-1])
    ch = np.concatenate(parts)
    sel = np.concatenate(picks)
    tt = (st - st[0]).astype("timedelta64[s]").astype(float) / 60.0
    tw = noise_model(st, tt, r[sel], raw[sel], ch, cfg,
                     None if state is None else np.asarray(state)[sel], exposure[sel])
    return tt, tw["z"], ch, tw["armed"], float(exposure[sel][tw["armed"]].sum()) / 1440.0, tw.get("nu", 200.0)


def pick_threshold(score, armed, worn_days, rate, max_occupancy=DEFAULTS["max_null_occupancy"], time=None, chunk=None):
    """Smallest threshold th such that `score >= th` fires at most `rate` events per worn
    day — at th and at every higher threshold — on eligible support. Unarmed rows
    and chunk/gap boundaries terminate events. A per-stream occupancy guard rejects
    almost-always-on solutions; this is a candidate engineering policy.

    Exact, not interpolated: an event starts at every up-crossing, and reading i is an
    up-crossing for exactly the thresholds in (score[i-1], score[i]], so the event count
    at every candidate threshold is two sorted searches. If even the highest threshold
    fires too often for the null's length, return a value just above the null's maximum
    (no events in the null: conservative)."""
    positive(worn_days=worn_days, rate=rate)
    if not 0 < max_occupancy <= 1:
        raise ValueError("max_occupancy must be in (0,1]")
    eligible = np.asarray(armed, bool)
    x = np.asarray(score, float)
    x = np.where(eligible & np.isfinite(x), x, -np.inf)
    fin = x[np.isfinite(x)]
    if fin.size == 0:
        return np.inf
    prev = np.r_[-np.inf, x[:-1]]
    if time is not None and len(x):
        boundary = np.r_[True, np.diff(time) >= GAP_THRESHOLD_MIN]
        if chunk is not None:
            boundary |= np.r_[True, np.diff(chunk) != 0]
        prev[boundary] = -np.inf
    up = x > prev
    lo, hi = np.sort(prev[up]), np.sort(x[up])
    cand = np.unique(fin)
    E = (len(hi) - np.searchsorted(hi, cand)) - (len(lo) - np.searchsorted(lo, cand))
    worst_above = np.maximum.accumulate(E[::-1])[::-1]
    occupancy = (len(np.sort(fin)) - np.searchsorted(np.sort(fin), cand)) / max(eligible.sum(), 1)
    good = np.flatnonzero((worst_above <= rate * worn_days) & (occupancy <= max_occupancy))
    return float(cand[good[0]]) if good.size else float(np.nextafter(fin.max(), np.inf))
