"""Smoke tests: the repo's own rules, enforced.

`python -m wearable selftest` is the same checks with readable output.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = ROOT / "wearable"


def test_selftest_passes():
    from wearable import selftest
    selftest.run()


def test_every_pool_discovers_methods():
    """Rule: adding a method means adding one file, nothing else."""
    from wearable import registry
    for pool, n_min in (("smoothing", 10), ("cpd", 4), ("models", 1), ("metrics", 6)):
        reg = registry.registry(pool)
        assert len(reg) >= n_min, f"{pool}: only {len(reg)} discovered"
        broken = {k: v.label for k, v in reg.items() if v.label.startswith("<unavail")}
        assert not broken, f"{pool}: failed to import {broken}"


def test_gap_threshold_has_one_definition():
    """The H8 threshold was once copied into 13 files. It must not drift back."""
    pattern = re.compile(r"^GAP_THRESHOLD_MIN(: *float)? *=", re.M)
    defining = [p for p in PKG.rglob("*.py") if pattern.search(p.read_text())]
    assert [p.name for p in defining] == ["config.py"], defining


def test_assign_chunks_has_one_definition():
    defining = [p for p in PKG.rglob("*.py") if "def assign_chunks" in p.read_text()]
    assert [p.name for p in defining] == ["chunking.py"], defining


def test_steps_do_not_import_each_other():
    """Rule: parts do not depend on each other; only the experiment runner connects them."""
    offenders = {}
    for p in (PKG / "steps").glob("*.py"):
        if p.name == "__init__.py":
            continue
        hits = re.findall(r"from wearable\.steps(?:\.(\w+))? import|"
                          r"import wearable\.steps\.(\w+)", p.read_text())
        others = {a or b for a, b in hits} - {"", p.stem}
        if others:
            offenders[p.name] = sorted(others)
    # join imports one helper from detect for the raw-only baseline; that is the only
    # allowance, and it is a pure reader with no step semantics.
    assert offenders in ({}, {"join.py": ["detect"]}), offenders


def test_split_is_independent_of_features():
    """Rule: comparisons keep the same test participants."""
    from wearable.steps.join import patient_split
    pats = [f"p{i:03d}" for i in range(100)]
    a = patient_split(pats, seed=42)
    b = patient_split(list(reversed(pats)), seed=42)
    assert a.equals(b)
    assert a.groupby("p_id")["split"].nunique().max() == 1


def test_cache_key_covers_everything_that_matters():
    """Rule: same thing twice reuses; anything different recomputes."""
    from wearable import store
    base = ("smooth", "m", {"a": 1}, [{"key": "k0"}], "h1")
    k = store.make_key(*base)
    assert k == store.make_key("smooth", "m", {"a": 1}, [{"key": "k0"}], "h1")
    assert k != store.make_key("smooth", "m", {"a": 2}, [{"key": "k0"}], "h1")
    assert k != store.make_key("smooth", "m", {"a": 1}, [{"key": "k1"}], "h1")
    assert k != store.make_key("smooth", "m", {"a": 1}, [{"key": "k0"}], "h2")
    assert k != store.make_key("detect", "m", {"a": 1}, [{"key": "k0"}], "h1")
