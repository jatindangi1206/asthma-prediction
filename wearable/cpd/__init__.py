"""Change-point detection pool. One file per method; drop one in and it is registered.

Contract — a module here is a CPD method if it exposes:

    detect(time, values, priors) -> {"anomaly", "phenotype", "magnitude", "severity"}
        per-row arrays; priors = {"m0", "sigma0"} from the per-patient noise floor

and optionally declares `LABEL`, `DEFAULTS` and `PHENOTYPES`. There is no list to edit
— see `wearable/registry.py`.

Detection runs on whichever column the detect step is pointed at — raw, smoothed or
residual — so a method can be used wherever it works best.

Determinism is each method's own responsibility: `kcpd` seeds
`np.random.default_rng(0)` and `hmm` passes `random_state=42`.
"""
from wearable.registry import available, get, registry

# The default ensemble, in vote order. Any discovered method can be used instead via
# the detect step's `methods` argument.
DEFAULT_ENSEMBLE = ("bocpd", "kcpd", "kalman", "hmm")


def all_methods():
    return registry("cpd")


def names():
    return available("cpd")


def method(name):
    return get("cpd", name)


def load_ensemble(keys=DEFAULT_ENSEMBLE):
    """-> [(name, module)] in the given order."""
    return [(k, get("cpd", k).load()) for k in keys]
