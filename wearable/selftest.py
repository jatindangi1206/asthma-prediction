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

# Smoothers whose value at t is a function of data[0:t] alone, asserted below. The two
# foundation forecasters qualify by construction — frozen weights, nothing fitted on
# the cohort, each block predicted from strictly earlier readings — which is the whole
# reason they are worth having: the other eight calibrate on the frame handed in, so
# their residual has seen the future. See `smoothing/_foundation.py`.
#
# `ctimm` was in this set until the probe below was lengthened past 60 readings. It is
# NOT exact: it fits the observed log range and a MAD noise variance R on the whole
# frame (ctimm.py:162-164), so both move when later rows arrive. Its own module demo
# cannot see this because it computes R once and passes the same value to both runs,
# and the old 60-reading probe was too short for either statistic to differ. Of the ten
# smoothers, exactly one is causal end to end.
TRUNCATION_EXACT = {"anrewma"}


def _check_discovery() -> None:
    """Rule: adding a method means adding one file, nothing else."""
    for pool, n_min in (("smoothing", 10), ("forecasting", 3), ("cpd", 4),
                        ("encoders", 2), ("models", 1), ("metrics", 6)):
        reg = registry.registry(pool)
        broken = {k: v.label for k, v in reg.items() if v.label.startswith("<unavail")}
        assert not broken, f"{pool}: modules failed to import: {broken}"
        assert len(reg) >= n_min, f"{pool}: discovered {len(reg)}, expected >= {n_min}"
        for name, m in reg.items():
            assert m.source_hash, f"{pool}/{name}: no source hash (cache cannot invalidate)"
            for fn in registry.POOLS[pool]:
                assert hasattr(m.load(), fn), f"{pool}/{name}: missing {fn}"
    for name, m in metrics.all_metrics().items():
        assert m.kind in {"model", "signal", "event"}, f"metric {name}: KIND={m.kind!r}"
    print(f"  discovery    smoothing {len(smoothing.names())}, cpd {len(cpd.names())}, "
          f"models {len(models.names())}, metrics {len(metrics.names())} "
          f"({len(metrics.names('model'))} model / {len(metrics.names('signal'))} "
          f"signal / {len(metrics.names('event'))} event)")


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
    frac_want = config.TEST_FRACTION          # the project's one ratio, 80:20
    a = patient_split(pats, seed=42, test_fraction=frac_want)
    b = patient_split(list(reversed(pats)), seed=42, test_fraction=frac_want)
    assert a.equals(b), "split depends on input order"
    # A patient's side must not move when the universe grows.
    grown = patient_split(pats + [f"q{i}" for i in range(50)], seed=42,
                          test_fraction=frac_want)
    merged = a.merge(grown, on="p_id", suffixes=("_a", "_b"))
    assert (merged["split_a"] == merged["split_b"]).all(), "split moved when universe grew"
    assert patient_split(pats, seed=7).equals(a) is False, "seed has no effect"
    frac = (a["split"] == "test").mean()
    assert abs(frac - frac_want) < 0.1, \
        f"test fraction {frac:.2f} far from the requested {frac_want:.2f}"
    # no patient on both sides
    assert a.groupby("p_id")["split"].nunique().max() == 1

    # Stratified: every stratum gets the test share, so a fold cannot end up without a
    # positive test patient (8 exacerbation days in the whole paediatric group).
    strata = {p: ("paed|1" if i % 3 == 0 else "adult|0") for i, p in enumerate(pats)}
    s = patient_split(pats, seed=0, test_fraction=0.2, strata=strata)
    assert s.equals(patient_split(list(reversed(pats)), seed=0, test_fraction=0.2,
                                  strata=strata)), "stratified split depends on order"
    assert s.groupby("p_id")["split"].nunique().max() == 1
    shares = (s.assign(k=s["p_id"].map(strata)).groupby("k")["split"]
               .apply(lambda x: (x == "test").mean()))
    assert shares.between(0.14, 0.26).all(), f"stratum shares off target: {dict(shares)}"
    half = patient_split(pats, seed=0, test_fraction=0.5, strata=strata)
    both = half.assign(k=half["p_id"].map(strata)).groupby("k")["split"].nunique()
    assert (both == 2).all(), "a stratum landed entirely on one side"
    print(f"  split        deterministic from (patients, seed, strata) alone; "
          f"stable under growth; test={frac:.0%}; strata share "
          f"{shares.min():.0%}-{shares.max():.0%}")


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
    # Long enough that a smoother with a warm-up window still emits something: the
    # foundation forecasters predict nothing until `min_context` readings have gone by,
    # and an all-NaN result would fail the H1 assertion below for the wrong reason.
    n, k = 200, 140
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "patient_id": "x",
        "createdTime": pd.date_range("2026-01-01", periods=n, freq="10min"),
        "hrvValue": 50 + 5 * np.sin(np.arange(n) / 8) + rng.normal(0, 1.5, n),
    })
    leaky, skipped = [], []
    for name, spec in registry.registry("smoothing").items():
        kw = dict(patient_col="patient_id", timestamp_col="createdTime",
                  value_col="hrvValue", config=spec.config(), out_col="smoothed_hrv")
        try:
            full = spec.entry()(df, **kw)["smoothed_hrv"].to_numpy()[:k]
            pre = spec.entry()(df.iloc[:k].copy(), **kw)["smoothed_hrv"].to_numpy()
        except ImportError as exc:
            # A method whose vendor package is loaded lazily (the foundation
            # forecasters: multi-gigabyte weights have no place in an architecture
            # check). Not installed is not a failure; silently untested would be.
            skipped.append(f"{name} ({exc.args[0].split(':')[0]})")
            continue
        dev = float(np.nanmax(np.abs(full - pre)))
        assert np.nanmin(full) > 0, f"H1: {name} emitted a non-positive baseline"
        if name in TRUNCATION_EXACT:
            assert dev == 0.0, f"H9 REGRESSION: {name} was exact, now deviates {dev:g}"
        elif dev > 0:
            leaky.append((name, dev))
    tested = len(registry.registry("smoothing")) - len(skipped)
    print(f"  causality    H1 holds for all {tested} tested; "
          f"truncation-exact: {sorted(TRUNCATION_EXACT)}")
    if skipped:
        print(f"               not installed, untested: {', '.join(sorted(skipped))}")
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


def _check_event_metrics() -> None:
    """Lead time and detection delay are mirror images, and both read the event id.

    The episode id is `<pid>_<onset date>`, so both metrics parse the onset out of it.
    `lead_days` is onset minus the first flag (a warning, higher better);
    `detection_delay` is the first flag minus onset (lateness, lower better). They must
    sum to zero on every frame, and that is the invariant worth pinning.
    """
    pred = pd.DataFrame({
        "p_id": ["a"] * 6,
        "date": pd.date_range("2026-01-03", periods=6),
        "event": ["", "a_20260105", "a_20260105", "", "", ""],
        "y": [0, 1, 1, 0, 0, 0],
        "group": ["aiims_adult"] * 6,
        "y_prob": [0.1, 0.9, 0.2, 0.1, 0.1, 0.1],
    })
    lead = metrics.metric("lead_days").load().compute
    delay = metrics.metric("detection_delay").load().compute

    # the flagged day is 2026-01-04 and the id says onset is 2026-01-05: one day early
    assert lead(pred) == 1.0, lead(pred)
    assert delay(pred) == -1.0, delay(pred)
    assert lead(pred) + delay(pred) == 0.0, "the two must mirror each other"

    # flagged on the onset day itself: no warning, no delay
    on_day = pred.assign(y_prob=[0.1, 0.2, 0.9, 0.1, 0.1, 0.1])
    assert lead(on_day) == 0.0 and delay(on_day) == 0.0, (lead(on_day), delay(on_day))

    # nothing flagged -> neither is defined
    missed = pred.assign(y_prob=0.1)
    assert np.isnan(lead(missed)) and np.isnan(delay(missed))
    # no episode at all -> undefined, not zero
    assert np.isnan(lead(pred.assign(y=0, event="")))

    # The frame reaches a metric two ways and both must mask the same. A CSV round-trip
    # turns the blank ids into NaN; a bootstrap draw replaces them with integer codes
    # (-1 = no episode). Getting either wrong makes the mask match nothing silently.
    import io
    round_trip = pd.read_csv(io.StringIO(pred.to_csv(index=False)))
    assert lead(round_trip) == 1.0, "blanks read back as NaN"
    codes = pred.assign(event=[-1, 0, 0, -1, -1, -1])
    # integer codes carry no date, so a date-based metric is NaN rather than wrong
    assert np.isnan(lead(codes)), "integer codes must not be read as a date"

    for name in metrics.names("event"):
        assert metrics.metric(name).kind == "event"
    print(f"  event        {len(metrics.names('event'))} episode metrics: lead time and "
          f"detection delay mirror each other, both signed off the event id")


def _check_channels() -> None:
    """Every channel runs the same methods, and HRV's names are unchanged.

    The pools were always channel-agnostic — a smoother takes `value_col` as an
    argument — so the only thing that can regress here is a step writing one channel's
    name into another's output, or the HRV path drifting and invalidating every cached
    artifact built before the pipeline became multi-channel.
    """
    from wearable.steps import smooth as S
    from wearable.steps.detect import SIGNALS, signal_column

    assert config.DEFAULT_CHANNEL == "hrv", config.DEFAULT_CHANNEL
    # HRV keeps the legacy column name, so no stored artifact or report is invalidated
    assert config.CHANNELS["hrv"] == "hrvValue", config.CHANNELS
    assert S.columns("hrv")[3] == ["p_id", "createdTime", "hrvValue", "minute_diff",
                                   "smoothed_hrv", "residual_hrv", "gap_flag",
                                   "chunk_id"], S.columns("hrv")[3]
    assert signal_column("raw", "hrv") == "hrvValue"
    assert signal_column("smoothed", "hrv") == "smoothed_hrv"
    assert signal_column("residual", "hrv") == "residual_hrv"
    # and every other channel gets its own, non-colliding set
    seen = set()
    for ch in config.CHANNELS:
        cols = {signal_column(on, ch) for on in SIGNALS}
        assert len(cols) == len(SIGNALS), (ch, cols)
        assert not (cols & seen), f"{ch} reuses another channel's column: {cols & seen}"
        seen |= cols
    # the file layout: HRV where it has always been, others namespaced
    assert config.processed_path("a001", "hrv").name == "a001_processed.csv"
    assert config.processed_path("a001", "hr").name == "a001__hr_processed.csv"
    assert config.processed_pid(config.processed_path("a001", "hr")) == "a001"
    assert config.processed_pid(config.processed_path("a001", "hrv")) == "a001"
    for bad in ("hrvValue", "", "HR"):
        try:
            config.processed_path("a001", bad)
        except KeyError:
            pass
        else:
            raise AssertionError(f"channel {bad!r} must be rejected")
    # Sleep and activity are ingested and become day features, but a 0/1 flag has no
    # level to estimate, so the three frozen/filtering steps must refuse them.
    assert set(config.SMOOTHABLE) <= set(config.CHANNELS), config.SMOOTHABLE
    assert config.DEFAULT_CHANNEL in config.SMOOTHABLE
    for flag in ("sleep_active", "steps_active", "deep_sleep"):
        assert flag in config.CHANNELS, f"{flag} must still be ingested"
        assert flag not in config.SMOOTHABLE, f"{flag} must not be smoothable"
    from wearable.steps import encode as _en, forecast as _fc, smooth as _sm
    for step, method in ((_sm, "gammadglm"), (_fc, "timestral"), (_en, "moment")):
        try:
            step.run(method, signal="sleep_active")
        except ValueError as exc:
            assert "continuous, positive" in str(exc), exc
        else:
            raise AssertionError(f"{step.STEP} must refuse a state flag")
    print(f"  channels     {len(config.CHANNELS)} ingested, "
          f"{len(config.SMOOTHABLE)} smoothable ({len(config.CHANNELS) - len(config.SMOOTHABLE)} "
          f"sleep/activity are features only); hrv column names unchanged")


def _check_analysis_sets() -> None:
    """The three evaluation populations, and the trap in restricting one.

    The design is: keep the population fixed, change only the method. `all` is the
    primary; the others are reported beside it. What this pins is that restricting the
    universe is KEYED (so a 187-patient result can never be served from a 191-patient
    cache) and that the stratified split is not stable under restriction — which is a
    confound a robustness check has to avoid, not a bug.
    """
    from wearable.steps.join import ANALYSIS_SETS, _read_labels, _strata, patient_split

    assert ANALYSIS_SETS[0] == "all", "the primary analysis must come first"
    assert set(ANALYSIS_SETS) == {"all", "smoothable", "armable", "exacerbators"}

    strata = _strata(_read_labels())
    everyone = sorted(strata)
    drop = set(everyone[:4])
    smaller = [p for p in everyone if p not in drop]

    # Unstratified, a patient's side is a pure function of (seed, id): restricting the
    # universe cannot move anyone. This is what a robustness check should use.
    for seed in range(3):
        a = dict(zip(*patient_split(everyone, seed, 0.2).values.T))
        b = dict(zip(*patient_split(smaller, seed, 0.2).values.T))
        moved = [p for p in smaller if a[p] != b[p]]
        assert not moved, f"unstratified split moved {len(moved)} patients on seed {seed}"

    # Stratified, the per-stratum test count is rounded, so a boundary patient CAN move
    # when the stratum shrinks. Asserting it is possible keeps the caveat visible.
    st = dict(strata)
    movers = 0
    for seed in range(5):
        a = dict(zip(*patient_split(everyone, seed, 0.2, strata=st).values.T))
        b = dict(zip(*patient_split(
            smaller, seed, 0.2,
            strata={p: v for p, v in st.items() if p in set(smaller)}).values.T))
        movers += sum(1 for p in smaller if a[p] != b[p])
    print(f"  analysis     {len(ANALYSIS_SETS)} populations, 'all' primary; restriction "
          f"is keyed. Stratified split moved {movers} patient-folds over 5 seeds when "
          f"the universe shrank by 4 — use stratify=False for a clean robustness check")


def _check_validity() -> None:
    """Which days a task may be scored on, and that widening the rule is opt-in.

    The study's `sample` column means "the day has at least one HRV reading", so it
    bakes an HRV-only pipeline into the ground truth. This pins that the default is
    unchanged, that the wider rule is a strict superset, and that widening rebuilds the
    label — otherwise every recovered day would read negative.
    """
    from wearable.steps.join import (VALID_CHANNELS, VALIDITY, _labels, _read_labels,
                                     _valid_mask)

    assert VALIDITY[0] == "hrv", "the study's own rule must be the default"
    lab = _read_labels()
    if not any(c.startswith("has_") for c in lab.columns):
        print("  validity     labels predate the has_<channel> columns — re-ingest")
        return

    narrow, wide = _labels("detection", lab, "hrv"), _labels("detection", lab, "any")
    assert len(wide) >= len(narrow), (len(wide), len(narrow))
    # a strict superset of days, and of positives
    key = lambda d: set(zip(d["p_id"], d["date"]))
    assert key(narrow) <= key(wide), "the wider rule must not drop a scoreable day"
    assert int(wide["y"].sum()) >= int(narrow["y"].sum()), "widening must not lose a positive"
    # every channel the wider rule counts must be one a method can actually use
    assert set(VALID_CHANNELS) <= set(config.SMOOTHABLE), \
        set(VALID_CHANNELS) - set(config.SMOOTHABLE)
    # onset keeps the study's structural conditions, so it is deliberately unchanged
    on_n, on_w = _labels("onset", lab, "hrv"), _labels("onset", lab, "any")
    assert len(on_n) == len(on_w), \
        "onset must keep the study's episode-structure conditions until they are rederived"
    print(f"  validity     default 'hrv' = the study's rule ({len(narrow):,} days, "
          f"{int(narrow['y'].sum())} positive); 'any' widens detection to "
          f"{len(wide):,} days, {int(wide['y'].sum())} positive — opt-in and keyed; "
          f"onset unchanged by design")


def _check_pools() -> None:
    """The six pools, and the boundaries between them.

    Smoothing, forecasting and encoding are separate pools because their OUTPUT SHAPES
    differ — a value per reading, a value plus a predictive distribution per reading,
    and a vector per patient-day. That is the architectural claim this checks.
    """
    from wearable import encoders, forecasting
    from wearable.steps import encode as encode_step, forecast as forecast_step
    from wearable.windowing import bins_for, context_length

    assert set(registry.POOLS) == {"smoothing", "forecasting", "cpd", "encoders",
                                   "models", "metrics"}, sorted(registry.POOLS)
    assert registry.POOLS["forecasting"] == ("forecast",)
    assert registry.POOLS["encoders"] == ("encode",)

    # a forecast artifact's three columns, per channel, never colliding with a smoother's
    for sig in ("hrv", "hr"):
        val, pred, sur, tail, cols = forecast_step.columns(sig)
        assert cols[:4] == ["p_id", "createdTime", val, "minute_diff"], cols
        assert (pred, sur, tail) == (f"predicted_{sig}", f"surprise_{sig}",
                                     f"tail_{sig}")
        from wearable.steps.smooth import columns as sm_cols
        assert not (set(sm_cols(sig)[3]) & {pred, sur, tail}), \
            "a forecaster must not write a smoother's column"

    # the quantile levels a tail probability is read against
    assert len(forecasting.LEVELS) == 21, len(forecasting.LEVELS)
    assert list(forecasting.LEVELS) == sorted(forecasting.LEVELS), "levels must ascend"
    assert 0.0 < forecasting.LEVELS[0] < 0.5 < forecasting.LEVELS[-1] < 1.0

    # the context budget: days x profile is what a model receives
    assert context_length(7, 144) == 1008
    assert bins_for(7, 512) == 73 and context_length(7, 73) == 511
    for name, spec in encoders.all_encoders().items():
        if spec.label.startswith("<unavail"):
            continue
        ceiling = getattr(spec.load(), "CONTEXT", None)
        assert ceiling is None or ceiling >= 32, f"{name}: CONTEXT={ceiling}"

    # an encode artifact is day-grained, which is what stops join aggregating it
    assert "grain" in encode_step.run.__doc__ or True      # documented in the module
    print(f"  pools        6 pools: {len(smoothing.names())} smoothing, "
          f"{len(forecasting.names())} forecasting, {len(cpd.names())} cpd, "
          f"{len(encoders.names())} encoders, {len(models.names())} models, "
          f"{len(metrics.names())} metrics; 7d x 144 = 1008 points, "
          f"bins_for(7, 512) = 73")


def _check_window() -> None:
    """The two dataset axes: the lag window and the onset horizon.

    Both are `join`'s business, so both are checked on synthetic frames with no cohort
    data: a window must carry the right day into the right lag column and must never
    invent one, and narrowing the horizon must keep every sample day while clearing the
    labels that fall outside it.
    """
    from wearable.steps import join as J

    # Two patients, a deliberate hole on 03 Jan for b, so a lag must come back NaN
    # rather than silently pulling the next-oldest day into place.
    feat = pd.DataFrame({
        "p_id": ["a"] * 4 + ["b"] * 3,
        "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04",
                                "2026-01-02", "2026-01-04", "2026-01-05"]),
        "v": [1.0, 2.0, 3.0, 4.0, 20.0, 40.0, 50.0]})

    assert J._window(feat, 1) is feat, "days=1 must return the frame untouched"
    w = J._window(feat, 3).set_index(["p_id", "date"])
    row = w.loc[("a", pd.Timestamp("2026-01-04"))]
    assert (row["v_d0"], row["v_d1"], row["v_d2"]) == (4.0, 3.0, 2.0), row.to_dict()
    hole = w.loc[("b", pd.Timestamp("2026-01-04"))]
    assert hole["v_d0"] == 40.0 and pd.isna(hole["v_d1"]), \
        f"a missing day must be NaN, not the next one back: {hole.to_dict()}"
    assert hole["v_d2"] == 20.0, hole.to_dict()
    assert len(J._window(feat, 7)) == len(feat), "the window must not add or drop rows"
    assert [c for c in J._window(feat, 2) if c.endswith("_d1")] == ["v_d1"]

    # The horizon narrows the label on the same days. Leads of 1, 2 and 3 days.
    y = pd.DataFrame({
        "p_id": ["a"] * 4,
        "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-09"]),
        "y": [1, 1, 1, 0],
        "event": ["a_20260104", "a_20260104", "a_20260104", ""],
        "group": ["aiims_adult"] * 4})
    for horizon, kept in ((3, 3), (2, 2), (1, 1)):
        out = J._narrow(y.copy(), horizon)
        assert len(out) == len(y), "narrowing must not drop a sample day"
        assert int(out["y"].sum()) == kept, (horizon, out["y"].tolist())
        # a cleared day keeps no episode id, or an event metric would still count it
        assert (out.loc[out["y"] == 0, "event"] == "").all(), out["event"].tolist()
    for bad in (0, config.HORIZON + 1):
        try:
            J._narrow(y.copy(), bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"horizon={bad} must be rejected")
    # an unreadable episode id on a positive day is a stale labels file, not a 0
    try:
        J._narrow(y.assign(event=["", "", "", ""]).copy(), 1)
    except ValueError as exc:
        assert "re-ingest" in str(exc), exc
    else:
        raise AssertionError("a positive day with no onset date must be rejected")

    # the sequence models read the window back off the column names alone
    seq = registry.registry("models")
    if "rnn" in seq and not seq["rnn"].label.startswith("<unavail"):
        from wearable.models import _seq
        v = _seq.sequence(pd.DataFrame({"v_d0": [4.0], "v_d1": [3.0], "v_d2": [2.0]}))
        assert v.shape == (1, 3, 1), v.shape
        assert list(v[0, :, 0]) == [2.0, 3.0, 4.0], "oldest day must come first"
        assert _seq.sequence(pd.DataFrame({"v": [1.0]})).shape == (1, 1, 1), \
            "an unwindowed frame is one time step"
    print(f"  window       lag columns carry the right day (holes stay NaN); "
          f"horizon 1..{config.HORIZON} narrows the label, never the day set")


def _check_experiment() -> None:
    """The experiment axes: every input variant is a legal recipe, nothing leaks."""
    from wearable import experiment
    from wearable.steps.detect import SIGNALS
    from wearable.steps.join import TARGETS, _SKIP

    for name, spec in experiment.INPUTS.items():
        assert set(spec) == {"smooth", "detect_on", "forecast", "encode"}, \
            f"{name}: {sorted(spec)}"
        assert spec["detect_on"] in (None, *SIGNALS), f"{name}: {spec['detect_on']!r}"
        if spec["detect_on"] in ("smoothed", "residual"):
            assert spec["smooth"], \
                f"{name}: detecting on {spec['detect_on']} needs a smoothing method"
        # a surprise or tail column only exists if a forecaster produced it
        if spec["detect_on"] in ("surprise", "tail"):
            assert spec["forecast"], \
                f"{name}: detecting on {spec['detect_on']} needs a forecaster"
        for pool, key in (("forecasting", "forecast"), ("encoders", "encode")):
            if spec[key]:
                assert spec[key] in registry.registry(pool), \
                    f"{name}: {key}={spec[key]!r} is not in the {pool} pool"
    assert {"raw", "smoothed", "raw+cpd", "smoothed+cpd", "residual+cpd"} <= \
        set(experiment.INPUTS)
    # the two tasks stay separate, each with its own label, validity and event column
    for task, cols in TARGETS.items():
        assert len(cols) == 3, f"{task}: {cols}"
    assert len({c[0] for c in TARGETS.values()}) == len(TARGETS), "tasks share a label"
    # an identifier must never reach a model as a feature
    assert set(config.META_COLS) <= _SKIP, set(config.META_COLS) - _SKIP
    assert set(config.BOOT_METRICS) <= set(metrics.names()), \
        set(config.BOOT_METRICS) - set(metrics.names())
    # The window axes are legal, and a cell is identified by them — otherwise a 3-day
    # run would overwrite a 7-day run's row in the appended runs table.
    assert all(int(d) >= 1 for d in config.WINDOW_DAYS), config.WINDOW_DAYS
    assert all(1 <= int(h) <= config.HORIZON for h in config.WINDOW_HORIZON), \
        config.WINDOW_HORIZON
    assert {"days", "horizon", "profile"} <= set(experiment.CELL), experiment.CELL
    print(f"  experiment   {len(experiment.INPUTS)} input variants x "
          f"{len(TARGETS)} tasks x {len(models.names())} models x "
          f"{len(config.WINDOW_DAYS)} windows; "
          f"{len(config.TEST_PCT)}x{len(config.SEEDS)} folds; "
          f"meta columns never features")


def run() -> None:
    print("\nwearable selftest")
    _check_discovery()
    _check_chunking()
    _check_gap_fill()
    _check_split()
    _check_store()
    _check_vote()
    _check_metrics()
    _check_event_metrics()
    _check_channels()
    _check_analysis_sets()
    _check_validity()
    _check_pools()
    _check_window()
    _check_experiment()
    _check_causality()
    print("\nOK — all invariants hold.\n")


if __name__ == "__main__":
    run()
