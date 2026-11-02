"""Step: join. Merge any results you already have into one modelling dataset, and split it.

    input   any number of smooth / detect artifacts (or none, for a raw-only baseline)
    output  tables `train` and `test`:
            p_id, date, <features from each input, per day of the window>, y, event, group
            plus `split` recording which patient went where

Each input artifact is aggregated to one row per patient-day and merged on (p_id, date).
Day-level ground truth comes from the labels file. Besides the label each row carries
its episode id (`event`, what an event metric scores) and its cohort group (`group`,
what a slice is reported by); `config.META_COLS` names them and no model trains on them.

**The two axes of the experiment live here, not in any model.** `days` is how much
history a row carries and `horizon` is how far ahead onset is asked about, and both are
properties of the dataset: a model is handed columns and never knows which axis
produced them. So the whole window ablation runs over the unchanged model pool.

    days=1 .. 7   the classified day plus the days-1 before it, as lag columns:
                  `<feature>_d0` is the classified day, `_d1` yesterday, `_d6` a week
                  back. A day with no recording gives NaN, which is informative (a
                  tree model learns which side it goes) and is not imputed away here.
    horizon=1..3  onset only: the same sample days, a narrower label. The labels hold
                  onset as "a new episode within 1-3 days"; `horizon=1` keeps the
                  positives whose episode starts the very next day and clears the rest.

Lags are built from EVERY day that has features, not only the days the target can
score, so the history of a scored day is complete even where the day before it is not
itself scoreable.

**The split is computed from the patient universe, the seed and the strata alone** —
never from feature content or from which artifacts were joined. So every variant in a
comparison sees exactly the same test participants, which is what makes the comparison
fair. The universe is every patient in the labels file, so a method that drops patients
yields fewer rows but cannot shift the split.

The strata and the universe come from the DETECTION labels whatever the target is, so
the two tasks are scored on the same test patients and can be read side by side.

Patients are split, never days: no patient appears in both train and test.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from wearable import store
from wearable.windowing import (ID as _WID, day_profile, lag_window,
                                profile_col)
from wearable.config import (BURN_IN_N, CHANNELS, DAYS, DEFAULT_CHANNEL,
                             DEFAULT_JOIN_CHANNELS, HORIZON, LABELS_CSV, META_COLS,
                             DURATION_CHANNELS, MIN_SEGMENT_ROWS, PROFILE, SEED,
                             SMOOTHABLE, STATE_CHANNELS,
                             TEST_FRACTION, processed_files, processed_pid)

STEP = "join"
# The window rule lives in `wearable/windowing.py` — one implementation, shared with
# the `encode` step. These aliases keep the old names working for callers and tests.
_window = lag_window
_profile = day_profile
TARGETS = {
    # name        -> (label column, validity column, event id column)
    "detection": ("label", "sample", "episode"),
    "onset": ("onset_label", "onset_valid", "onset_id"),
}

# WHICH DAYS A TASK MAY BE SCORED ON.
#
# The study's own `sample` column means "the day has at least one HRV reading" (raw
# README §3), which bakes an HRV-only pipeline into the ground truth: a day carrying
# 710 heart-rate readings and no HRV is marked unscoreable, and so is the episode it
# belongs to. That was right when HRV was the only channel. It no longer is.
#
#   hrv   the study's own rule. THE DEFAULT, so every stored result keeps its meaning.
#   any   the day holds a reading in any channel a method can actually use. Measured
#         against the study's rule on this cohort: days 15,691 -> 20,050 (+28%),
#         exacerbation days 144 -> 204 (+42%), episodes 57 -> 73 (+28%), patients
#         191 -> 221 (the 30 with heart rate but no surviving HRV).
#
# Switching is a decision about what the paper claims, not a bug fix, so it is a keyed
# argument and never the default. `label` is `exac_day` restricted to valid days, so
# widening validity widens the positive class with it.
VALIDITY = ("hrv", "any")
VALID_CHANNELS = ("hrv", "hr", "temp", "steps")   # what `any` counts as usable
_ID = ["p_id", "date"]
# Never aggregated as features: identifiers, strings, and the labels themselves.
_SKIP = {"p_id", "date", "createdTime", "chunk_id", "label", "sample", "exac_day",
         "episode", "onset_valid", "onset_label", "onset_id", *META_COLS}


def _uniform(seed: int, pid: str) -> float:
    """Stable uniform in [0,1) from (seed, pid) — the one source of split randomness."""
    return int(hashlib.sha256(f"{seed}:{pid}".encode()).hexdigest()[:16], 16) / 16 ** 16


def patient_split(patients, seed: int = SEED, test_fraction: float = TEST_FRACTION,
                  strata=None):
    """Deterministic patient-level split from the id list, the seed and the strata only.

    Hashing each id with the seed (rather than shuffling) means a patient's side is
    stable even if the universe changes — adding a cohort cannot move existing
    patients between train and test.

    `strata` (a dict or Series pid -> stratum) draws the test share *within* each
    stratum, ranking its members by the same hash: with ~1% positive days and only 8
    exacerbation days among the children, an unstratified draw can leave a fold with no
    positive test patient at all and no computable AUROC. Stratifying costs a little of
    the growth stability above — a patient at a stratum's rounding boundary can move
    when that stratum gains members — but the split is still a pure function of
    (universe, seed, fraction, strata), which is what a fair comparison needs.
    """
    ids = sorted(set(map(str, patients)))
    u = {p: _uniform(seed, p) for p in ids}
    if strata is None:
        side = {p: "test" if u[p] < test_fraction else "train" for p in ids}
    else:
        strata = dict(strata)
        buckets: dict[str, list[str]] = {}
        for p in ids:
            buckets.setdefault(str(strata.get(p, "")), []).append(p)
        side = {}
        for key in sorted(buckets):
            members = sorted(buckets[key], key=lambda p: (u[p], p))
            n_test = int(len(members) * test_fraction + 0.5)      # rounded half up
            side.update({p: ("test" if i < n_test else "train")
                         for i, p in enumerate(members)})
    return pd.DataFrame({"p_id": ids, "split": [side[p] for p in ids]})


def _day_features(table: pd.DataFrame, prefix: str,
                  profile: int = 0) -> pd.DataFrame:
    """One row per patient-day. Numeric columns -> mean/std/min/max; `*_anomaly` flags
    -> rows flagged plus distinct events (contiguous runs), which is the honest way to
    count a sustained-regime detector. With `profile` > 0, also that many time-of-day
    slots of the raw signal, so a sequence model can see the day itself."""
    t = table.copy()
    t["createdTime"] = pd.to_datetime(t["createdTime"])
    t["date"] = t["createdTime"].dt.normalize()

    num = [c for c in t.columns
           if c not in _SKIP and pd.api.types.is_numeric_dtype(t[c])]
    # A 0/1 state is counted, not averaged: total minutes in it, and how many separate
    # bouts. The anomaly flags already worked this way, so they share the path.
    flags = [c for c in num if c.endswith("_anomaly") or c == "anomaly_present"
             or c in STATE_CHANNELS]
    # Sleep stages arrive already totalled per bout ("45 min deep, 20 light"), with no
    # start or end for any stage. So a day gets the SUM and the bout count. Averaging
    # bout durations would answer a question nobody asked, and spreading them over a
    # timeline would invent one the export does not contain.
    durations = [c for c in num if c in DURATION_CHANNELS]
    plain = [c for c in num if c not in flags and c not in durations]

    out = []
    if plain:
        agg = t.groupby(_ID)[plain].agg(["mean", "std", "min", "max"])
        agg.columns = [f"{prefix}_{c}_{s}" for c, s in agg.columns]
        out.append(agg)
    if durations:
        g = t.groupby(_ID)[durations]
        tot = g.sum(min_count=1)
        tot.columns = [f"{prefix}_{c}_minutes" for c in tot.columns]
        cnt = g.count()
        cnt.columns = [f"{prefix}_{c}_bouts" for c in cnt.columns]
        out.extend([tot, cnt])
    if flags:
        g = t.sort_values(["p_id", "createdTime"]).groupby(_ID)
        fl = g[flags].sum()
        fl.columns = [f"{prefix}_{c}_minutes" for c in fl.columns]
        ev = g[flags].agg(lambda s: int(((s.to_numpy() == 1) &
                                        (np.r_[0, s.to_numpy()[:-1]] == 0)).sum()))
        ev.columns = [f"{prefix}_{c}_bouts" for c in flags]
        out.extend([fl, ev])
    n = t.groupby(_ID).size().rename(f"{prefix}_n_rows").to_frame()
    out.append(n)
    if profile:
        prof = _profile(t, prefix, int(profile))
        if prof is not None:
            out.append(prof)
    return pd.concat(out, axis=1).reset_index()


def _features(artifacts, profile: int = 0,
              channels=DEFAULT_JOIN_CHANNELS) -> pd.DataFrame:
    """Every input's day features side by side, one row per patient-day.

    Built independently of the labels so the window below can reach days the target
    cannot score. Artifacts are outer-merged: a day one method dropped is a NaN row for
    that method, never a missing day.
    """
    parts = []
    for a in artifacts:
        tbl = a.read_table(a.manifest["tables"][0])
        tbl["p_id"] = tbl["p_id"].astype(str)
        prefix = f"{a.step[:2]}_{a.method}".replace("+", "_")
        if a.manifest.get("grain") == "day":
            # Already one row per patient-day (an `encode` artifact). Aggregating it
            # would turn a d-wide embedding into 4d columns, three quarters of them a
            # std/min/max over a single row — and `_day_features` needs a
            # `createdTime` these rows do not have, so it would raise, not degrade.
            tbl["date"] = pd.to_datetime(tbl["date"]).dt.normalize()
            cols = [c for c in tbl.columns if c not in _ID]
            parts.append(tbl[[*_ID, *cols]].rename(
                columns={c: f"{prefix}_{c}" for c in cols}))
            continue
        parts.append(_day_features(tbl, prefix=prefix, profile=profile))
    # EVERY MODEL GETS EVERY VARIABLE. A path decides what happens to the channel it
    # processes; every other channel still goes in, raw. So the raw blocks are always
    # added — minus the channels an artifact already covers, which would otherwise
    # appear twice under two prefixes.
    covered = {a.manifest.get("stats", {}).get("signal", DEFAULT_CHANNEL)
               for a in artifacts}
    from wearable.steps.detect import _load_input   # the one allowed cross-step import
    for ch in channels:
        if ch in covered:
            continue
        if not processed_files(ch):
            print(f"  [join] no {ch} files — run `ingest` to include it")
            continue
        raw = _load_input(None, "raw", signal=ch)
        raw["p_id"] = raw["p_id"].astype(str)
        # Profile only the continuous channels: a day profile of a 0/1 state is 144
        # copies of the state, which says nothing its daily total does not.
        parts.append(_day_features(
            raw, f"raw_{ch}" if ch != DEFAULT_CHANNEL else "raw",
            profile=profile if ch in SMOOTHABLE else 0))
    if not parts:
        raise FileNotFoundError(
            f"nothing to join: no artifacts and no ingested files for {list(channels)}")
    feat = parts[0]
    for part in parts[1:]:
        feat = feat.merge(part, on=_ID, how="outer")
    return feat


# The three analysis populations the evaluation design uses. `all` is the default and
# the primary; the other two are reported beside it, never instead of it.
#
#   all           every patient the labels know about — the primary analysis
#   smoothable    those with at least one unbroken chunk long enough to smooth
#                 (config.MIN_SEGMENT_ROWS). A robustness check, not a filter to
#                 prefer: all ten smoothers agree on this set, and the patients it
#                 drops are all non-exacerbators.
#   armable       those with enough valid samples for CPD to arm (config.BURN_IN_N)
#   exacerbators  those with any exacerbation day. SELECTION ON THE OUTCOME, so it is
#                 a clearly-labelled secondary analysis answering a different
#                 question — "among patients who do exacerbate, can we separate their
#                 bad days from their good ones?" — and never the headline. It
#                 inflates the base rate ~3x and makes the false-alarm rate
#                 unmeasurable in the population you would deploy to.
ANALYSIS_SETS = ("all", "smoothable", "armable", "exacerbators")


def analysis_set(name: str = "all", channel: str = DEFAULT_CHANNEL) -> list[str]:
    """-> the patient ids of one named analysis population, sorted."""
    if name not in ANALYSIS_SETS:
        raise ValueError(f"analysis set must be one of {list(ANALYSIS_SETS)}, "
                         f"got {name!r}")
    lab = _read_labels()
    y_col, valid_col, _ = TARGETS["detection"]
    d = lab[lab[valid_col] == 1]
    everyone = sorted(set(d["p_id"].astype(str)))
    if name == "all":
        return everyone
    if name == "exacerbators":
        pos = d.groupby("p_id")[y_col].max()
        return sorted(str(p) for p in pos[pos == 1].index)

    # Both data-quality sets are derived from the ingested series, not from any
    # artifact, so they do not depend on which method happened to be run.
    from wearable.chunking import assign_chunks
    keep = []
    for f in processed_files(channel):
        pid = processed_pid(f)
        col = CHANNELS[channel]
        df = pd.read_csv(f, usecols=["createdTime", col])
        obs = df[pd.to_numeric(df[col], errors="coerce").notna()].copy()
        if name == "armable":
            if len(obs) >= BURN_IN_N:
                keep.append(pid)
            continue
        if obs.empty:
            continue
        obs["createdTime"] = pd.to_datetime(obs["createdTime"])
        obs["p_id"] = pid
        chunks = assign_chunks(obs, "createdTime", "p_id")
        if int(chunks["chunk_id"].value_counts().max()) >= MIN_SEGMENT_ROWS:
            keep.append(pid)
    return sorted(set(keep) & set(everyone))


def _read_labels() -> pd.DataFrame:
    if not LABELS_CSV.exists():
        raise FileNotFoundError(
            f"{LABELS_CSV} not found — run ingest with a labelled source first")
    return pd.read_csv(LABELS_CSV, parse_dates=["date"],
                       dtype={"p_id": str, "episode": str, "onset_id": str,
                              "group": str})


def _valid_mask(lab: pd.DataFrame, target: str, valid: str) -> pd.Series:
    """Which rows this target may be scored on, under the chosen validity rule."""
    y_col, valid_col, _ = TARGETS[target]
    if valid == "hrv":
        return lab[valid_col] == 1
    have = [f"has_{c}" for c in VALID_CHANNELS if f"has_{c}" in lab.columns]
    if not have:
        raise ValueError(
            "the labels file has no `has_<channel>` columns, so validity cannot be "
            "widened — re-run `python -m wearable ingest` to record them")
    any_reading = lab[have].fillna(0).astype(int).max(axis=1) == 1
    if target == "detection":
        return any_reading
    # Onset keeps the study's other two conditions — not inside an episode, and not
    # among the patient's last 3 days — because those are about the EPISODE structure,
    # not about which channel was recorded. Only the data-availability half widens.
    return any_reading & (lab[valid_col] == 1)


def _labels(target: str, lab: pd.DataFrame, valid: str = "hrv") -> pd.DataFrame:
    """The days this target can score, with the label, its episode id and the group."""
    y_col, valid_col, event_col = TARGETS[target]
    out = lab[_valid_mask(lab, target, valid)].copy()
    if valid != "hrv" and target == "detection":
        # `label` is `exac_day` on the study's sample days, so on a widened day set the
        # label has to be rebuilt from `exac_day` or every recovered day reads negative.
        out[y_col] = out["exac_day"].fillna(0).astype(int)
    out["y"] = out[y_col].astype(int)
    out["event"] = out[event_col].fillna("").astype(str)
    out["group"] = out["group"].fillna("") if "group" in out.columns else ""
    return out[list(META_COLS)]


def _narrow(y: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Onset within fewer days: the same sample days, only the label narrows.

    The labels hold onset as "a new episode within 1 to `config.HORIZON` days", and the
    event id is `<pid>_<onset date>` naming the NEAREST onset — so a day whose nearest
    onset falls outside the horizon has no onset inside it, and both its label and its
    event id go. Keeping the day set fixed is what lets two horizons be compared: the
    denominator is the same, only the question got stricter.
    """
    if not 1 <= horizon <= HORIZON:
        raise ValueError(f"horizon must be 1..{HORIZON} (what the labels hold), "
                         f"got {horizon}")
    y = y.copy()                               # never write into the caller's frame
    pos = y["y"] == 1
    onset = pd.to_datetime(y["event"].str.rsplit("_", n=1).str[-1],
                           format="%Y%m%d", errors="coerce")
    lead = (onset - y["date"]).dt.days
    off = pos & ~lead.between(1, HORIZON)      # NaN compares False, so it lands here
    if off.any():
        raise ValueError(
            f"{int(off.sum())} onset-positive days have no readable onset date 1-"
            f"{HORIZON} days ahead in `event` (e.g. {y.loc[off, 'event'].iloc[0]!r} on "
            f"{y.loc[off, 'date'].iloc[0].date()}) — the labels file predates the "
            f"episode-id format; re-ingest")
    drop = pos & (lead > horizon)
    y.loc[drop, "event"] = ""
    y.loc[drop, "y"] = 0
    return y


def _strata(lab: pd.DataFrame) -> dict:
    """pid -> "<group>|<has an exacerbation day>", from the DETECTION labels.

    Taken from detection whatever the target is, so detection and onset are scored on
    the same test patients (the reference design's rule).
    """
    y_col, valid_col, _ = TARGETS["detection"]
    d = lab[lab[valid_col] == 1]
    group = d.groupby("p_id")["group"].first() if "group" in d.columns else None
    pos = d.groupby("p_id")[y_col].max().astype(int)
    return {p: f"{'' if group is None else group.get(p, '')}|{int(pos.get(p, 0))}"
            for p in pos.index}


def run(artifacts=(), target: str = "detection", seed: int = SEED,
        test_fraction: float = TEST_FRACTION, force: bool = False,
        name: str | None = None, stratify: bool = True,
        days: int = DAYS, horizon: int = HORIZON, profile: int = PROFILE,
        channels=DEFAULT_JOIN_CHANNELS, patients=None, valid: str = "hrv"):
    """`patients` restricts the analysis population — the universe the split is drawn
    from. Use `analysis_set(name)` to name one. Omitted, every patient in the labels is
    analysed, which is the primary design: keep the population fixed, change only the
    method."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {sorted(TARGETS)}, got {target!r}")
    if horizon != HORIZON and target != "onset":
        raise ValueError(f"horizon is an onset setting; target={target!r} is asked "
                         f"about the day itself, so its lead is always 0")
    channels = tuple(channels) if channels else DEFAULT_JOIN_CHANNELS
    unknown = set(channels) - set(CHANNELS)
    if unknown:
        raise ValueError(f"unknown channels {sorted(unknown)}; "
                         f"available: {sorted(CHANNELS)}")
    if valid not in VALIDITY:
        raise ValueError(f"valid must be one of {list(VALIDITY)}, got {valid!r}")
    artifacts = list(artifacts)

    # The artifact's name describes the combination, so variants sit side by side.
    name = name or ("+".join(f"{a.step}:{a.method}" for a in artifacts) or "raw_only")
    params = {"target": target, "seed": seed, "test_fraction": test_fraction,
              "n_inputs": len(artifacts), "stratify": bool(stratify),
              "days": int(days), "horizon": int(horizon),
              "profile": int(profile)}
    if patients is not None:
        # The population is part of what a result means, so it is keyed — a 187-patient
        # run can never be served from a 191-patient cache.
        ids = sorted(set(map(str, patients)))
        params = {**params, "n_patients_requested": len(ids),
                  "patients_sha": hashlib.sha256(",".join(ids).encode()
                                                 ).hexdigest()[:12]}
    # Keyed only when it is not the single-channel default, so every join artifact
    # built before the pipeline became multi-channel keeps its key and is reused.
    if tuple(channels) != tuple(DEFAULT_JOIN_CHANNELS):
        params = {**params, "channels": list(channels)}
    if valid != "hrv":
        params = {**params, "valid": valid}
    inputs = [a.ref for a in artifacts]
    code_hash = "join3"            # this step's own logic version

    if not force:
        hit = store.find(STEP, name, params, inputs, code_hash)
        if hit is not None:
            print(f"[join/{name}] reusing {hit.key[:12]}")
            return hit

    lab = _read_labels()
    y = _labels(target, lab, valid)
    if horizon != HORIZON:
        y = _narrow(y, horizon)
    # Universe and strata = every patient the DETECTION labels know about, independent
    # of the target, the inputs and both window axes, so every variant, both tasks and
    # every window share the split.
    strata = _strata(lab)
    if patients is not None:
        wanted = set(map(str, patients))
        missing = wanted - set(strata)
        if missing:
            raise ValueError(f"{len(missing)} requested patients are not in the "
                             f"labels (e.g. {sorted(missing)[:3]})")
        strata = {p: v for p, v in strata.items() if p in wanted}
        if not strata:
            raise ValueError("the requested analysis set is empty")
    split = patient_split(strata, seed, test_fraction,
                          strata=strata if stratify else None)

    # Features first, then the window, then the labels: the lag columns of a scored day
    # may come from days the target cannot score.
    frame = y.merge(lag_window(_features(artifacts, profile, channels), days),
                    on=_ID, how="left")
    frame = frame.merge(split, on="p_id", how="left")
    frame = frame[frame["split"].notna()]

    art = store.open_artifact(STEP, name, params, inputs, code_hash, label=name)
    feature_cols = [c for c in frame.columns
                    if c not in {*META_COLS, "split"}]
    for part in ("train", "test"):
        art.write_table(part, frame[frame["split"] == part]
                        .drop(columns="split").reset_index(drop=True))
    art.write_table("split", split)

    tr = frame[frame["split"] == "train"]
    te = frame[frame["split"] == "test"]
    events = lambda d: int(d.loc[d["y"] == 1, "event"].replace("", np.nan).nunique())
    art.manifest["feature_cols"] = feature_cols
    art.complete(target=target, days=int(days), horizon=int(horizon),
                 profile=int(profile), channels=list(channels), valid=valid,
                 n_universe=len(strata),
                 n_features=len(feature_cols),
                 n_train=len(tr), n_test=len(te),
                 n_train_patients=tr["p_id"].nunique(),
                 n_test_patients=te["p_id"].nunique(),
                 train_positives=int(tr["y"].sum()), test_positives=int(te["y"].sum()),
                 train_events=events(tr), test_events=events(te),
                 test_groups=sorted(set(te["group"]) - {""}))
    print(f"\n[join/{name}] target={target}, {days}-day window"
          + (f", {profile} slots/day" if profile else "")
          + (f", onset within {horizon} day{'s' * (horizon > 1)}"
             if target == "onset" else ""))
    print(f"  train {len(tr):,} days / {tr['p_id'].nunique()} patients "
          f"({int(tr['y'].sum())} positive, {events(tr)} episodes)")
    print(f"  test  {len(te):,} days / {te['p_id'].nunique()} patients "
          f"({int(te['y'].sum())} positive, {events(te)} episodes)")
    print(f"  {len(feature_cols)} features, key {art.key[:12]}")
    return art
