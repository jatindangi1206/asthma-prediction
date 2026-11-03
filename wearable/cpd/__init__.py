"""Change-point detection pool. One file per method; drop one in and it is registered.

Contract — a module here is a CPD method if it exposes:

    detect(time, values, priors, **settings) -> {"anomaly", "phenotype", "magnitude", "severity"}
        per-row arrays over ONE patient's observed readings, time-sorted.
        time    minutes since the patient's first reading (clock time, gaps included)
        values  the whitened residual z from `_whiten.noise_model` (NaN before arming)
        priors  {"thresholds": {stream: value}, "chunk": H8 ids, "stamp": datetime64,
                 "raw": the channel's own value, "step_min": nominal cadence,
                 "nu": the Student-t tail of the patient's whitened values,
                 "armed": where an alarm can be reported (burn-in and settle are not)}
        settings  the method's DEFAULTS, overridden per channel by config.CPD_SETTINGS

and optionally declares `LABEL`, `DEFAULTS`, `PHENOTYPES`, and for calibration either
`score(time, values, priors, **settings) -> {stream: per-row statistic}` (thresholds are
then set on simulated no-change noise) or `threshold_for(rate, **settings)` (an analytic
threshold). There is no list to edit — see `wearable/registry.py`. A module starting
with `_` is a helper (`_whiten`), never a method.

Detection runs on whichever column the detect step is pointed at — raw, smoothed,
residual or surprise — after the noise model has whitened it.

Determinism is each method's own responsibility: `kcpd` seeds
`np.random.default_rng(0)`; the others are deterministic.
"""
from wearable.registry import available, get, registry

# The default ensemble, in vote order. Any discovered method can be used instead via
# the detect step's `methods` argument.
DEFAULT_ENSEMBLE = ("bocpd", "kcpd", "kalman", "hmm", "drift")


def all_methods():
    return registry("cpd")


def names():
    return available("cpd")


def method(name):
    return get("cpd", name)


def load_ensemble(keys=DEFAULT_ENSEMBLE):
    """-> [(name, module)] in the given order."""
    return [(k, get("cpd", k).load()) for k in keys]
