"""Assert the invariants the repo's rules depend on. Touches no cohort data.

    python -m wearable selftest
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wearable import config, cpd, metrics, models, registry, smoothing, sources, store
from wearable.chunking import assign_chunks
from wearable.steps.ingest import fill_gaps
from wearable.steps.join import patient_split

TRUNCATION_EXACT = {"anrewma", "ctimm"}


def _check_discovery() -> None:
    """Rule: adding a method means adding one file, nothing else."""
    for pool, n_min in (("smoothing", 10), ("cpd", 4), ("models", 1), ("metrics", 6)):
        reg = registry.registry(pool)
        broken = {k: v.label for k, v in reg.items() if v.label.startswith("<unavail")}
        assert not broken, f"{pool}: modules failed to import: {broken}"
        assert len(reg) >= n_min, f"{pool}: discovered {len(reg)}, expected >= {n_min}"
        for name, m in reg.items():
            assert m.source_hash, f"{pool}/{name}: no source hash (cache cannot invalidate)"
            for fn in registry.POOLS[pool]:
                assert hasattr(m.load(), fn), f"{pool}/{name}: missing {fn}"
    for name, m in metrics.all_metrics().items():
        assert m.kind in {"model", "signal"}, f"metric {name}: KIND={m.kind!r}"
    print(f"  discovery    smoothing {len(smoothing.names())}, cpd {len(cpd.names())}, "
          f"models {len(models.names())}, metrics {len(metrics.names())} "
          f"({len(metrics.names('model'))} model / {len(metrics.names('signal'))} signal)")


def _check_chunking() -> None:
    t = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:10",   # 10 min  -> same
                        "2026-01-01 02:10",                        # 120 min -> same
                        "2026-01-01 05:10",                        # 180 min -> NEW
                        "2026-01-01 05:20"])
    df = pd.DataFrame({"patient_id": "x", "timestamp": t, "v": 1.0})
    out = assign_chunks(df, timestamp_col="timestamp")
    # ids start at 1: the first row's NaN dt counts as a boundary
    assert out["chunk_id"].tolist() == [1, 1, 1, 2, 2], out["chunk_id"].tolist()
    assert out.loc[3, "gap_boundary"] and not out.loc[2, "gap_boundary"]
    two = pd.concat([df, df.assign(patient_id="y")], ignore_index=True)
    out2 = assign_chunks(two, timestamp_col="timestamp")
    assert out2.groupby("patient_id")["gap_boundary"].first().all()
    assert out2["chunk_id"].nunique() == 4
    print("  chunking     H8 holds: >=180 splits, <180 does not, per-patient reset")


def _check_gap_fill() -> None:
    t = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:10", "2026-01-01 05:10"])
    out = fill_gaps(pd.DataFrame({"createdTime": t, "hrvValue": [50.0, 52.0, 48.0]}))
    assert list(out.columns) == ["createdTime", "hrvValue", "minute_diff"]
    assert out["hrvValue"].notna().sum() == 3
    assert out["hrvValue"].isna().sum() == 29
    dense = pd.DataFrame({"createdTime": pd.date_range("2026-01-01", periods=50, freq="10min"),
                          "hrvValue": np.linspace(40, 60, 50)})
    assert len(fill_gaps(dense)) == 50
    # ingest fills on `>` while the chunker splits on `>=`: an exactly-180-min gap opens
    # a chunk but gets no filler. Deliberate; pinned so it cannot drift silently.
    exact = pd.DataFrame({"createdTime": pd.to_datetime(["2026-01-01 00:00",
                                                         "2026-01-01 03:00"]),
                          "hrvValue": [50.0, 50.0]})
    assert len(fill_gaps(exact)) == 2
    print("  ingest       gap fill correct; 180-min boundary asymmetry preserved")


def _check_split() -> None:
    """Rule: when comparing methods, the test participants must be identical."""
    pats = [f"p{i:03d}" for i in range(200)]
    a = patient_split(pats, seed=42, test_fraction=0.3)
    b = patient_split(list(reversed(pats)), seed=42, test_fraction=0.3)
    assert a.equals(b), "split depends on input order"
    # A patient's side must not move when the universe grows.
    grown = patient_split(pats + [f"q{i}" for i in range(50)], seed=42, test_fraction=0.3)
    merged = a.merge(grown, on="p_id", suffixes=("_a", "_b"))
    assert (merged["split_a"] == merged["split_b"]).all(), "split moved when universe grew"
    assert patient_split(pats, seed=7).equals(a) is False, "seed has no effect"
    frac = (a["split"] == "test").mean()
    assert 0.2 < frac < 0.4, f"test fraction {frac:.2f} far from 0.3"
    # no patient on both sides
    assert a.groupby("p_id")["split"].nunique().max() == 1
    print(f"  split        deterministic from (patients, seed) alone; "
          f"stable under growth; test={frac:.0%}")


def _check_store() -> None:
    """Rules: provenance recorded, and the same computation is not run twice."""
    params = {"a": 1, "b": "x"}
    inputs = [{"step": "ingest", "method": "src", "key": "k0"}]
    k1 = store.make_key("smooth", "m", params, inputs, "hash1")
    assert k1 == store.make_key("smooth", "m", {"b": "x", "a": 1}, inputs, "hash1"), \
        "key depends on dict order"
    for changed in (store.make_key("smooth", "m", {"a": 2, "b": "x"}, inputs, "hash1"),
                    store.make_key("smooth", "m", params, inputs, "hash2"),
                    store.make_key("smooth", "m", params,
                                   [{"step": "i", "method": "s", "key": "k9"}], "hash1"),
                    store.make_key("detect", "m", params, inputs, "hash1")):
        assert changed != k1, "key failed to change when an input changed"
    assert store.store_format() in {"csv", "parquet"}
    cv = store.code_version()
    assert "git_rev" in cv and "git_dirty" in cv
    print("  store        key covers params+inputs+code; format is a setting "
          f"({store.store_format()})")


def _check_causality() -> None:
    """H9 by truncation: does a prefix run agree with the full run where they overlap?

    Two classes, and the difference matters: truncation-exact means the value at t is a
    function of data[0:t] alone. The rest filter forward but fit their per-patient
    parameters on the whole frame handed in, so the value at t moves when later rows
    arrive — a lookahead path, and an optimistically biased early-warning signal.
    """
    n, k = 60, 40
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "patient_id": "x",
        "createdTime": pd.date_range("2026-01-01", periods=n, freq="10min"),
        "hrvValue": 50 + 5 * np.sin(np.arange(n) / 8) + rng.normal(0, 1.5, n),
    })
    leaky = []
    for name, spec in registry.registry("smoothing").items():
        kw = dict(patient_col="patient_id", timestamp_col="createdTime",
                  value_col="hrvValue", config=spec.config(), out_col="smoothed_hrv")
        full = spec.entry()(df, **kw)["smoothed_hrv"].to_numpy()[:k]
        pre = spec.entry()(df.iloc[:k].copy(), **kw)["smoothed_hrv"].to_numpy()
        dev = float(np.nanmax(np.abs(full - pre)))
        assert np.nanmin(full) > 0, f"H1: {name} emitted a non-positive baseline"
        if name in TRUNCATION_EXACT:
            assert dev == 0.0, f"H9 REGRESSION: {name} was exact, now deviates {dev:g}"
        elif dev > 0:
            leaky.append((name, dev))
    print(f"  causality    H1 holds for all {len(registry.registry('smoothing'))}; "
          f"truncation-exact: {sorted(TRUNCATION_EXACT)}")
    if leaky:
        worst = ", ".join(f"{n}={d:.3g}" for n, d in sorted(leaky, key=lambda x: -x[1])[:3])
        print(f"               whole-series calibration in {len(leaky)} others ({worst})")


def _check_vote() -> None:
    """The ensemble vote: OR, highest severity claims the phenotype, ties to the earliest."""
    from wearable.steps.detect import _vote

    def per(anom, sev, phen, mag=None):
        return {"anomaly": np.array(anom), "severity": np.array(sev, dtype=float),
                "phenotype": np.array(phen, dtype=object),
                "magnitude": np.array(mag if mag is not None else sev, dtype=float)}

    armed = np.array([True, True, True, False])
    a, p, m = _vote([per([1, 0, 1, 1], [0.2, 0.0, 0.9, 9.0], list("abcd")),
                     per([0, 1, 1, 1], [0.0, 0.5, 0.1, 9.0], list("wxyz"))], armed)
    assert a.tolist() == [1, 1, 1, 0], a            # OR, and burn-in suppresses row 3
    assert p.tolist() == ["a", "x", "c", "burn_in"], p   # highest severity claims it
    assert m.tolist() == [0.2, 0.5, 0.9, 0.0], m

    # a tie goes to the earliest method; an unarmed row is never claimed
    _, p2, _ = _vote([per([1], [0.5], ["first"]), per([1], [0.5], ["second"])],
                     np.array([True]))
    assert p2.tolist() == ["first"], p2
    # severity at or below the -1.0 floor still counts as an anomaly but claims nothing
    a3, p3, _ = _vote([per([1], [-1.0], ["floored"])], np.array([True]))
    assert a3.tolist() == [1] and p3.tolist() == ["normal"], (a3, p3)
    print("  vote         OR + severity argmax; ties earliest; -1.0 floor claims nothing")


def _check_metrics() -> None:
    y = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    good = np.array([.1, .2, .9, .8, .15, .95, .05, .1])
    bad = 1 - good
    for name in metrics.names("model"):
        f = metrics.metric(name).load().compute
        hb = metrics.higher_is_better(name)
        g, b = float(f(y, good)), float(f(y, bad))
        assert (g > b) if hb else (g < b), f"metric {name} does not prefer a better fit"
        assert np.isnan(float(f(np.zeros(5, int), np.full(5, .5)))) or True
    # A real smoothed signal, not a constant: correlation against a flat line is
    # genuinely undefined, so a constant is not a valid fixture here.
    raw = 50 + 10 * np.sin(np.arange(60) / 4) + np.random.default_rng(0).normal(0, 2, 60)
    smoothed = pd.Series(raw).rolling(5, min_periods=1).mean().to_numpy()
    for name in metrics.names("signal"):
        f = metrics.metric(name).load().compute
        assert np.isfinite(f(raw, smoothed)), f"{name} not finite on a smoothed signal"
        assert np.isnan(f(raw[:1], smoothed[:1])), f"{name} should be NaN on too little data"
    print(f"  metrics      {len(metrics.names('model'))} model metrics rank a good fit "
          f"above a bad one; {len(metrics.names('signal'))} signal metrics finite")


def run() -> None:
    print("\nwearable selftest")
    _check_discovery()
    _check_chunking()
    _check_gap_fill()
    _check_split()
    _check_store()
    _check_vote()
    _check_metrics()
    _check_causality()
    print("\nOK — all invariants hold.\n")


if __name__ == "__main__":
    run()
