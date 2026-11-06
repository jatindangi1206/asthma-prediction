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

Night D publishes at the first observation at/after its common noon decision only
if that observation is eligible and less than one day late. No verdict is moved
through settling. Held magnitudes expire after a day; alarms are explicit impulses.

The threshold is analytic, not simulated: u_D is ~N(0,1) per night under no change, and
Siegmund's approximation gives the CUSUM's average run length, so `threshold_for` solves
for the h that fires `rate` times per worn day (≈ per night).
"""

import numpy as np
from scipy.optimize import brentq
from wearable.cpd._contracts import series, positive, excess

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
    positive(rate=rate, cusum_k=p["cusum_k"])
    if p["direction"] not in (-1, 0, 1):
        raise ValueError("direction must be -1, 0 or 1")
    arl = 1.0 / max(rate, 1e-9) * (1 if p["direction"] else 2)   # two-sided: each side half
    if _siegmund(p["cusum_k"], 0.01) >= arl:
        return {"level": 0.01}
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
    h = np.timedelta64(60, "m")
    out = {}
    if not len(day):
        return out
    for d in np.arange(day[0], day[-1] + np.timedelta64(2, "D")):
        decision = d + int(p["window_end_h"]) * h
        if decision > stamp[-1]:
            continue
        lo = d - np.timedelta64(1, "D") + int(p["window_start_h"]) * h
        # Select the rule for THIS completed night, never from a future sleep flag.
        has_sleep = state is not None and np.any(
            (stamp >= lo) & (stamp < decision) & (np.asarray(state) == _SLEEP))
        if has_sleep:
            hi = decision
            m = ok & (stamp >= lo) & (stamp < hi) & (np.asarray(state) == _SLEEP)
        else:
            hi = d + 6 * h
            m = ok & (stamp >= d) & (stamp < hi)
        if m.sum() * step < p["min_sleep_minutes"]:
            continue
        after = int(np.searchsorted(stamp, decision))
        # Do not move an old verdict through arbitrary nonwear or settling. An
        # expired/unarmed night may inform the baseline but emits no verdict.
        eligible = (after < len(stamp) and stamp[after] < decision + np.timedelta64(1, "D")
                    and (armed is None or armed[after]))
        out[d] = (float(np.median(raw[m])), after if eligible else -1)
    return out


def _scores(time, values, priors=None, **params):
    p = {**DEFAULTS, **params}
    pr = priors or {}
    series(time, values, pr)
    positive(baseline_nights=p["baseline_nights"], min_nights=p["min_nights"],
             min_sleep_minutes=p["min_sleep_minutes"], cusum_k=p["cusum_k"])
    if p["direction"] not in (-1, 0, 1):
        raise ValueError("invalid drift direction")
    if (p["min_nights"] > p["baseline_nights"] or
            any(int(p[k]) != p[k] for k in ("min_nights", "baseline_nights", "window_start_h", "window_end_h")) or
            not 0 <= p["window_end_h"] < p["window_start_h"] < 24):
        raise ValueError("drift requires integer night counts and a cross-midnight hour window")
    n = len(values)
    out = np.full(n, np.nan)
    verdict = np.zeros(n, bool)
    stamp = pr.get("stamp")
    if stamp is None:
        return out, verdict
    raw = np.asarray(pr.get("raw", values), float)
    step = float(pr.get("step_min") or 1.0)
    th = {**THRESHOLDS, **pr.get("thresholds", {})}["level"]
    armed = pr.get("armed")
    nights = _nights(np.asarray(stamp, "datetime64[ns]"), raw, step, p,
                     None if armed is None else np.asarray(armed, bool), pr.get("state"))
    dates = sorted(nights)
    up = dn = 0.0
    base, last_i, last_val = [], None, np.nan
    published = np.full(n, np.nan)
    previous_day = None
    for d in dates:
        level, i = nights[d]
        if previous_day is not None and d - previous_day > np.timedelta64(1, "D"):
            up = dn = 0.0
        previous_day = d
        if i < 0:
            base.append((d, level))
            continue
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
            up = dn = 0.0
        if np.isfinite(val) and val >= th:
            # Signal, then restart the CUSUM (the standard procedure): a shift that
            # persists re-alarms every few nights instead of holding one alarm for
            # weeks — measured, the no-restart version flagged 69% of HR time.
            up = dn = 0.0
            verdict[i] = True
        else:
            base.append((d, level))                  # flagged nights stay out of baseline
        if last_i is not None and i > last_i:
            out[last_i:i] = last_val
            published[last_i:i] = time[last_i]
        last_i, last_val = i, val
    if last_i is not None:
        out[last_i:] = last_val
        published[last_i:] = time[last_i]
    out[np.asarray(time) - published >= 1440.0] = np.nan
    return out, verdict


def score(time, values, priors=None, **params):
    return {"level": _scores(time, values, priors, **params)[0]}


def detect(time, values, priors=None, **params):
    n = len(values)
    out = {"anomaly": np.zeros(n, int),
           "phenotype": np.full(n, "normal", dtype=object),
           "magnitude": np.zeros(n), "severity": np.zeros(n)}
    if n < 2:
        return out
    s, hit = _scores(time, values, priors, **params)
    th = {**THRESHOLDS, **(priors or {}).get("thresholds", {})}["level"]
    # A drift alarm is a once-a-night verdict, not a state: flag the reading where it
    # arrives; magnitude and severity keep holding through the day as features.
    out["anomaly"] = hit.astype(int)
    out["phenotype"] = np.where(hit, PHENOTYPE, "normal").astype(object)
    out["magnitude"] = s
    out["severity"] = excess(s, th)
    return out


# --- registry metadata (auto-discovered; see wearable/registry.py) ---
LABEL = "Nightly-level CUSUM drift"
PHENOTYPES = (
    'Drift_Nightly_Level',
)
