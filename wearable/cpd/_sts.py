"""Structural time-series (STS) simulation: a second way to build the no-change data the
detect step calibrates its thresholds on. A `_` helper, not a detector.

The default calibration (`_whiten.twin`) resamples whole days of the patient's own past.
This one instead learns a model of the patient's normal behaviour from the same past and
simulates new data from it. Both are run through the identical pipeline afterwards
(smoother -> residual -> binning -> noise model -> detectors), and each threshold is set
so complete alarm events occur at the target rate. `config.CPD_CALIBRATION` picks one.

THE MODEL (per patient, per channel, refitted at each recalibration on past data only)

    log y_t = mu + circadian_t + AR_t + b_sleep*S_t + b_active*A_t + e_t

    mu           fixed intercept                              the patient's level
    circadian_t  trigonometric seasonal, period 24 h,         24-h rhythm
                 `harmonics` pairs (deterministic)
    AR_t         AR(2) at the channel's cadence               autocorrelation: a fast
                                                              root (minutes) and a slow
                                                              one (hours) — slow baseline
                                                              variation that reverts
    S_t, A_t     sleep / active indicators (rest = 0)         state shifts, from the
                                                              ingested flags
    e_t          irregular, Student-t                         measurement noise

It runs on a regular grid at the channel's cadence (HR 1 min, HRV 10 min). Each reading
is placed at its nearest grid cell, and every cell without a reading is NaN, which the
Kalman filter handles natively. Physiology runs on clock time, so the grid is clock
time; nothing is resampled or interpolated.

WHY THESE CHOICES, AND WHERE THEY ARE WEAK

* Stationary, not a random-walk level. The obvious structural model (local level +
  stochastic seasonal) was tried first and rejected on the data: on 1-minute HR the
  fitted level variance absorbs activity swings (~6e-5 per minute), so a simulated week
  wanders to a log-sd of ~0.8 — 50 to 190 bpm against a real 80 to 115. Real HR reverts
  to its daily pattern; a random walk never does. AR(2) keeps slow variation over hours
  but mean-reverts. Limitation: day-to-day baseline variation beyond hours is not
  modelled (the smoother removes it from the residual, and `drift` is calibrated
  analytically, so the detectors calibrated here are not the ones that need it).
* Log scale: HR and HRV are positive and their spread grows with their level. A fit on
  the raw scale could simulate non-positive values the smoothers cannot take (H1).
* Heavy tails are kept, not removed. Nothing beyond k MADs is dropped, because normal
  extremes and the onset of a real shift cannot be told apart that way. The state shocks
  are Gaussian, but the irregular is drawn from a Student-t whose nu is fitted to the
  standardised one-step forecast errors (the HR residual has excess kurtosis ~9; a
  Gaussian would make no-change data too tame and the thresholds too low). Limitation:
  HR's 1–5-min surges are bursts, not independent noise; AR(1) plus t noise only
  approximates them.
* Simulation in independent blocks. Each block is a fresh draw from the fitted model's
  end state, `fit_days` long, so no single path runs for 60 days. Blocks sit one empty
  day apart, so the smoother and the noise model reset between
  them (H8), as they would for a real gap of that length.
* The patient's own observation mask. A block's readings fall exactly where the history
  window's readings fell, shifted by whole days so clock times are kept, and carry the
  same sleep / active states. Gaps, wear patterns and missingness are the patient's own.
* Fallback for short history. Fewer than `min_days` distinct days of data cannot separate
  a 24-h rhythm from the level, so the circadian term is dropped (the "reduced" model).
  If even that fit fails, the caller falls back to day shuffling for that recalibration
  and records it. At arming only the first worn day exists, so the first recalibration
  is always reduced.
* Device limits: values are clipped to `[lo, hi]` (HRV is capped at 129 by the device).
  The model cannot reproduce HRV's pile-up at 120–129, which is a device behaviour, not
  a Gaussian-ish one.
* Constant state offsets: a sleep shift is one number. In reality it depends on how deep
  and how long the sleep is.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import t as student_t

DEFAULTS = {
    "grid_min": 10.0,     # the model's time step: the channel's cadence
    "harmonics": 2,       # trigonometric pairs of the 24-h seasonal
    "fit_days": 7.0,      # history the model is fitted on (also the block length)
    "min_days": 3.0,      # distinct days of data needed for the circadian term
    "lo": 1.0,            # device / physiological limits for the simulated values
    "hi": 1.0e9,
    "maxiter": 60,        # optimiser iterations for the maximum-likelihood fit
}
_MIN_OBS = 60             # below this many readings, no model is fitted
_DAY = np.timedelta64(1, "D")


def _minutes(stamp, origin):
    return (stamp - origin).astype("timedelta64[s]").astype(float) / 60.0


def fit(stamp, value, state, cfg):
    """Fit the model on past readings (time-sorted). -> a dict, or None if it cannot.

    stamp: datetime64[ns]; value: the channel's own value; state: _whiten state codes.
    """
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    from wearable.cpd._whiten import ACTIVE, SLEEP

    ok = np.isfinite(value) & (value > 0)
    stamp, value, state = stamp[ok], value[ok], np.asarray(state)[ok]
    if value.size < _MIN_OBS:
        return None
    step = float(cfg["grid_min"])
    origin = stamp[0]
    j = np.round(_minutes(stamp, origin) / step).astype(int)
    n = int(j[-1]) + 1
    y = np.full(n, np.nan)
    y[j] = np.log(value)                               # last reading in a cell wins
    n_days = np.unique(stamp.astype("datetime64[D]")).size
    seasonal = n_days >= cfg["min_days"]
    # state indicators, only where the history actually shows both sides of them
    cols, X = [], np.zeros((n, 0))
    for code, name in ((SLEEP, "sleep"), (ACTIVE, "active")):
        on = state == code
        if 30 <= on.sum() <= on.size - 30:
            x = np.zeros(n)
            x[j[on]] = 1.0
            X = np.column_stack([X, x])
            cols.append(name)
    # stationary by construction: fixed intercept, deterministic 24-h shape, AR(2)
    kw = dict(level=True, stochastic_level=False, autoregressive=2, irregular=True,
              exog=X if cols else None)
    if seasonal:
        kw.update(freq_seasonal=[{"period": 1440.0 / step, "harmonics": int(cfg["harmonics"])}],
                  stochastic_freq_seasonal=[False])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = UnobservedComponents(y, **kw).fit(disp=False, method="lbfgs",
                                                    maxiter=int(cfg["maxiter"]))
            e = np.asarray(res.standardized_forecasts_error[0])
    except Exception:
        return None
    e = e[np.isfinite(e)][res.loglikelihood_burn:]
    nu = float(np.clip(student_t.fit(e, floc=0.0)[0], 3.0, 200.0)) if e.size > 30 else 200.0
    params = dict(zip(res.model.param_names, map(float, res.params)))
    return {"res": res, "model": "full" if seasonal else "reduced", "nu": nu,
            "exog": cols, "origin": origin, "end": origin + np.timedelta64(int((n - 1) * step * 60), "s"),
            "step": step, "params": params, "n_obs": int(value.size), "n_days": int(n_days)}


def simulate(f, hist_stamp, hist_state, worn_needed_min, cfg, seed=0):
    """No-change synthetic readings on the history's own observation mask.

    Blocks of the history window are laid end to end, one empty day apart; within each,
    readings fall at the history's own times (shifted by whole days) with its own
    states, and their values are a fresh draw from the fitted model's end state.
    -> (stamp, value, state), time-sorted, at least `worn_needed_min` worn minutes.
    """
    from wearable.cpd._whiten import ACTIVE, SLEEP

    res, step = f["res"], f["step"]
    rs = np.random.RandomState(seed)
    end = f["end"]
    hist_stamp = np.asarray(hist_stamp, "datetime64[ns]")
    hist_state = np.asarray(hist_state)
    span = int(np.ceil((end - hist_stamp[0]) / _DAY)) or 1      # whole days per block
    shift = span * _DAY
    # where each history reading lands in a block: index j of the simulation grid,
    # whose step 0 is one step after the end of the fitted sample
    pos = np.round(_minutes(hist_stamp + shift, end) / step).astype(int) - 1
    keep = (pos >= 0)
    pos, base_stamp, base_state = pos[keep], hist_stamp[keep] + shift, hist_state[keep]
    if pos.size == 0:
        return np.array([], "datetime64[ns]"), np.array([]), np.array([], int)
    nsim = int(pos.max()) + 1
    X = None
    if f["exog"]:
        X = np.zeros((nsim, len(f["exog"])))
        for c, name in enumerate(f["exog"]):
            code = SLEEP if name == "sleep" else ACTIVE
            X[pos[base_state == code], c] = 1.0
    sig_e = np.sqrt(max(f["params"].get("sigma2.irregular", 0.0), 0.0))
    nu = f["nu"]
    unit_t = np.sqrt((nu - 2) / nu) if nu > 2 else 1.0
    out_s, out_v, out_k = [], [], []
    worn, k = 0.0, 0
    while worn < worn_needed_min:
        shocks = rs.standard_t(nu, size=(nsim, 1)) * unit_t * sig_e
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = np.asarray(res.simulate(nsim, anchor="end", exog=X,
                                          measurement_shocks=shocks,
                                          random_state=rs)).reshape(-1)
        val = np.clip(np.exp(sim[pos]), cfg["lo"], cfg["hi"])
        out_s.append(base_stamp + k * (span + 1) * _DAY)      # one empty day between blocks
        out_v.append(val)
        out_k.append(base_state)
        worn += pos.size * float(cfg.get("reading_min", step))
        k += 1
    return np.concatenate(out_s), np.concatenate(out_v), np.concatenate(out_k)


def report(f):
    """The fitted parameters, flattened for the calibration table."""
    if f is None:
        return {}
    p = {f"sts_{k.replace('.', '_').replace('(', '_').replace(')', '').replace(',', '_')}": v
         for k, v in f["params"].items()}
    return {"sts_model": f["model"], "sts_nu": f["nu"], "sts_n_obs": f["n_obs"],
            "sts_n_days": f["n_days"], "sts_exog": "+".join(f["exog"]) or "none", **p}
