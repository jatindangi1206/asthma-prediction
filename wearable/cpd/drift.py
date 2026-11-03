"""Slow drift in the NIGHTLY level of the channel itself — the multi-day prodrome.

Every residual-based detector is blind to change slower than its smoother's memory
(about 3 h for anrewma, about 17 h for gammadglm), and an exacerbation builds over days.
This one reads the channel's own value (`priors["raw"]`, the ingested signal, not the
residual) and asks whether tonight's resting level has moved away from the patient's
own recent sleep:

    level_D = median of the SLEEP readings of the night ending on the morning of D
              (window 18:00 on D-1 to 12:00 on D; a night needs `min_sleep_minutes`
              of sleep worn; without sleep flags, the clock night 00:00–06:00)
    u_D     = (level_D - median of the baseline nights) / (1.4826 · MAD of them)
    S+_D    = max(0, S+_{D-1} + u_D - k),  S-_D = max(0, S-_{D-1} - u_D - k)
    score   = S+ (direction +1), S- (direction -1), or max(S+, S-) (direction 0)

The baseline is the trailing `baseline_nights` nights before D, skipping nights the
detector itself flagged, so a real decline cannot become the new normal; it needs
`min_nights` of them before the detector arms. After a signal the CUSUM restarts from
zero, so each alarm covers one night and a lasting shift alarms again a few nights on.
Sleep is the most comparable stretch of the day (no activity, posture or meals), and
asthma worsens at night.

Night D is only known once its window closes, so its score is set at the first armed
reading at or after 12:00 on D and held until the next night's; the alarm itself is
flagged on that one reading. Causal.

The threshold is analytic, not simulated: u_D is ~N(0,1) per night under no change, and
Siegmund's approximation gives the CUSUM's average run length, so `threshold_for` solves
for the h that fires `rate` times per worn day (≈ per night).
"""

import numpy as np
from scipy.optimize import brentq

_SLEEP = 1                       # cpd/_whiten.py's SLEEP state code

DEFAULTS = {
    "window_start_h": 18,        # the night ending on D starts at 18:00 on D-1 ...
    "window_end_h": 12,          # ... and ends at 12:00 on D
    "baseline_nights": 28,
    "min_nights": 3,
    "min_sleep_minutes": 60.0,   # worn sleep minutes a night needs to have a level
    "direction": 0,              # +1 rises only, -1 falls only, 0 both
    "cusum_k": 0.5,              # slack in baseline sds (tuned to a ~1-sd shift)
}
THRESHOLDS = {"level": 5.0}      # fallback without calibration
PHENOTYPE = "Drift_Nightly_Level"


def _siegmund(k, h):
    b = h + 1.166
    return (np.exp(2 * k * b) - 2 * k * b - 1) / (2 * k * k)


def threshold_for(rate, **params):
    """h such that the CUSUM fires `rate` times per night on N(0,1) nightly levels."""
    p = {**DEFAULTS, **params}
    arl = 1.0 / max(rate, 1e-9) * (1 if p["direction"] else 2)   # two-sided: each side half
    return {"level": float(brentq(lambda h: _siegmund(p["cusum_k"], h) - arl, 0.01, 200))}


def _nights(stamp, raw, step, p, armed=None, state=None):
    """-> {date D: (level, index of the first ARMED reading at or after the window end)}.

    The night ending on the morning of D is the window [D-1 window_start_h, D
    window_end_h). Its level is the median of the readings flagged SLEEP in it, so a
    night is always compared with the patient's other sleep — wherever the sleep fell
    on the clock. Without flags (state None or all unknown) it falls back to the clock
    night 00:00–06:00 of D. An unarmed reading (burn-in, a settle window) would
    suppress the verdict, so it lands on the first armed one."""
    day = stamp.astype("datetime64[D]")
    ok = np.isfinite(raw)
    has_sleep = state is not None and np.any(np.asarray(state) == _SLEEP)
    h = np.timedelta64(60, "m")
    out = {}
    for d in np.unique(day):
        if has_sleep:
            lo = d - np.timedelta64(1, "D") + int(p["window_start_h"]) * h
            hi = d + int(p["window_end_h"]) * h
            m = ok & (stamp >= lo) & (stamp < hi) & (np.asarray(state) == _SLEEP)
        else:
            hi = d + 6 * h
            m = ok & (stamp >= d) & (stamp < hi)
        if m.sum() * step < p["min_sleep_minutes"]:
            continue
        after = np.searchsorted(stamp, hi)
        if armed is not None:
            ok_after = np.flatnonzero(armed[after:])
            after = after + int(ok_after[0]) if ok_after.size else len(stamp)
        if after < len(stamp):
            out[d] = (float(np.median(raw[m])), int(after))
    return out


def score(time, values, priors=None, **params):
    p = {**DEFAULTS, **params}
    pr = priors or {}
    n = len(values)
    out = np.full(n, np.nan)
    stamp = pr.get("stamp")
    if stamp is None:
        return {"level": out}
    raw = np.asarray(pr.get("raw", values), float)
    step = float(pr.get("step_min") or 1.0)
    th = {**THRESHOLDS, **pr.get("thresholds", {})}["level"]
    armed = pr.get("armed")
    nights = _nights(np.asarray(stamp, "datetime64[ns]"), raw, step, p,
                     None if armed is None else np.asarray(armed, bool), pr.get("state"))
    dates = sorted(nights)
    up = dn = 0.0
    base, last_i, last_val = [], None, np.nan
    for d in dates:
        level, i = nights[d]
        recent = [(dd, lv) for dd, lv in base
                  if (d - dd).astype("timedelta64[D]").astype(int) <= p["baseline_nights"]]
        if len(recent) >= p["min_nights"]:
            lv = np.array([x for _, x in recent])
            med = np.median(lv)
            sd = max(1.4826 * np.median(np.abs(lv - med)), 1.0)    # 1 device unit floor
            u = (level - med) / sd
            up = max(0.0, up + u - p["cusum_k"])
            dn = max(0.0, dn - u - p["cusum_k"])
            val = {1: up, -1: dn}.get(int(p["direction"]), max(up, dn))
        else:
            val = np.nan
        if np.isfinite(val) and val >= th:
            # Signal, then restart the CUSUM (the standard procedure): a shift that
            # persists re-alarms every few nights instead of holding one alarm for
            # weeks — measured, the no-restart version flagged 69% of HR time.
            up = dn = 0.0
        else:
            base.append((d, level))                  # flagged nights stay out of baseline
        if last_i is not None and i > last_i:
            out[last_i:i] = last_val
        last_i, last_val = i, val
    if last_i is not None:
        out[last_i:] = last_val
    return {"level": out}


def detect(time, values, priors=None, **params):
    n = len(values)
    out = {"anomaly": np.zeros(n, int),
           "phenotype": np.full(n, "normal", dtype=object),
           "magnitude": np.zeros(n), "severity": np.zeros(n)}
    if n < 2:
        return out
    s = np.nan_to_num(score(time, values, priors, **params)["level"], nan=0.0)
    th = {**THRESHOLDS, **(priors or {}).get("thresholds", {})}["level"]
    # A drift alarm is a once-a-night verdict, not a state: flag the reading where it
    # arrives; magnitude and severity keep holding through the day as features.
    held = s >= th
    hit = held & ~np.r_[False, held[:-1]]
    out["anomaly"] = hit.astype(int)
    out["phenotype"] = np.where(hit, PHENOTYPE, "normal").astype(object)
    out["magnitude"] = s
    out["severity"] = np.clip(s / (2.0 * th), 0.0, 1.0)
    return out


# --- registry metadata (auto-discovered; see wearable/registry.py) ---
LABEL = "Nightly-level CUSUM drift"
PHENOTYPES = (
    'Drift_Nightly_Level',
)
