"""Step: change-point detection. Runs CPD methods on whichever signal you point it at.

    input   `on="raw"`                  -> the ingest artifact (data/processed/)
            `on="smoothed"|"residual"`  -> a smooth artifact
            `on="surprise"|"tail"`      -> a forecast artifact
    output  one long table `detected`:
            p_id, createdTime, <signal>, chunk_id, armed,
            <method>_anomaly / _phenotype / _magnitude  for each method run,
            anomaly_present, primary_phenotype, magnitude,
            noise_z, noise_scale              the noise model's whitened value and scale
            reentry_jump_z, reentry_gap_min   a level change across a gap, set once the
                                              settle window after it has passed

Every detector reads the column through a per-patient noise model (`cpd/_whiten.py`)
that whitens it, arms after the patient's first worn day, and calibrates each
detector's threshold on no-change noise built from that day. All of it is causal.

Some methods work better on a raw signal and some on a de-trended one, so the input
column is an argument, not an architecture. `residual` keeps the dual-track rationale
(detection on raw false-alarms on circadian drift; on smoothed it loses the volatility
bursts) while remaining one option among three.

Depends on no other step: it is handed an artifact and a column name.
"""
from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

from wearable import cpd, registry, store
from wearable.config import (CHANNELS, CPD_BUDGET, CPD_CALIBRATION, CPD_CALIBRATIONS,
                             CPD_SETTINGS, DEFAULT_CHANNEL, STS_SETTINGS,
                             GAP_THRESHOLD_MIN, PROCESSED_DIR, processed_files,
                             processed_path, processed_pid)
from wearable.cpd import _sts, _whiten

warnings.filterwarnings("ignore")

STEP = "detect"
# Column templates, not fixed names: the same three choices apply to every channel.
# `{val}` is the channel's own value column (`config.CHANNELS`), `{sig}` its key — so
# for HRV these resolve to exactly the names this step has always used.
SIGNALS = {"raw": "{val}", "smoothed": "smoothed_{sig}", "residual": "residual_{sig}",
           # from a `forecast` artifact: the departure from a frozen model's
           # expectation, and that departure as a calibrated tail probability
           "surprise": "surprise_{sig}", "tail": "tail_{sig}"}


def signal_column(on: str, signal: str = DEFAULT_CHANNEL) -> str:
    """Which column `--on <on>` reads, for a given channel."""
    if on not in SIGNALS:
        raise ValueError(f"on must be one of {sorted(SIGNALS)}, got {on!r}")
    if signal not in CHANNELS:
        raise ValueError(f"signal must be one of {sorted(CHANNELS)}, got {signal!r}")
    return SIGNALS[on].format(val=CHANNELS[signal], sig=signal)


def cpd_settings(signal: str, methods) -> tuple[dict, dict]:
    """-> (the step's cold-start knobs, {method: its settings}) for one channel.

    Each detector's DEFAULTS, overridden by that channel's block in
    `config.CPD_SETTINGS`. A name the detector does not have is an error, not a silent
    no-op: a misspelt threshold would otherwise run the defaults and look tuned.
    """
    table = CPD_SETTINGS.get(signal, {})
    unknown = set(table) - {"detect"} - set(registry.registry("cpd"))
    if unknown:
        raise ValueError(f"CPD_SETTINGS[{signal!r}]: unknown detectors {sorted(unknown)}")
    step = dict(_whiten.DEFAULTS)
    for name, over, known in [("detect", table.get("detect", {}), step)] + [
            (m, table.get(m, {}), registry.get("cpd", m).defaults) for m in methods]:
        bad = set(over) - set(known)
        if bad:
            raise ValueError(f"CPD_SETTINGS[{signal!r}][{name!r}]: unknown settings "
                             f"{sorted(bad)}; available: {sorted(known)}")
    step.update(table.get("detect", {}))
    return step, {m: registry.get("cpd", m).params(**table.get(m, {})) for m in methods}


def _ensure_chunk_id(df: pd.DataFrame) -> pd.DataFrame:
    if "chunk_id" in df.columns and df["chunk_id"].notna().any():
        return df
    dt = df["createdTime"].diff().dt.total_seconds() / 60.0
    df["chunk_id"] = (dt.isna() | (dt >= GAP_THRESHOLD_MIN)).cumsum().astype(int)
    return df


def _vote(per: list[dict], armed: np.ndarray):
    """OR the detectors; report the firing one with the highest severity.

    ponytail: vectorised — the row-by-row version was 4 passes over 2.9M rows in pure
    Python. argmax takes the first maximum, which is the earliest method, matching the
    strict `>` of the loop it replaced. The `> -1.0` floor matches it too: a detector
    that fires with a severity at or below that still counts toward `anomaly_present`
    but cannot claim the phenotype.
    """
    anom = np.stack([p["anomaly"] for p in per]) == 1
    sev = np.stack([p["severity"] for p in per])
    fired = anom & armed
    claims = fired & (sev > -1.0)
    has = claims.any(0)
    best = np.where(claims, sev, -np.inf).argmax(0)

    primary = np.where(armed, "normal", "burn_in").astype(object)
    mag = np.zeros(anom.shape[1])
    rows = np.flatnonzero(has)
    if rows.size:
        pick = best[rows]
        primary[rows] = np.stack([p["phenotype"] for p in per])[pick, rows]
        mag[rows] = np.stack([p["magnitude"] for p in per])[pick, rows]
    return fired.any(0).astype(int), primary, mag


def _blank(n):
    return {"anomaly": np.zeros(n, int), "phenotype": np.full(n, "normal", dtype=object),
            "magnitude": np.zeros(n), "severity": np.zeros(n)}


def _prepare(stamp, t, r, raw, chunk, state, cfg):
    """The series the noise model and detectors run on: the readings themselves, or
    their median bins when `bin_min` > 0 (HR). Shared by the real data and by STS
    synthetic data, so both take the identical path. -> (series, cfg, holder, last)."""
    state = state if state is not None else np.zeros(len(t), int)
    if cfg["bin_min"] > 0:
        ser = _whiten.bins(stamp, t, r, raw, chunk, cfg["bin_min"], state)
        return (ser, {**cfg, "step_min": max(cfg["step_min"], cfg["bin_min"])},
                ser["holder"], ser["last"])
    ser = {"stamp": stamp, "t": t, "r": r, "raw": raw, "chunk": chunk, "state": state}
    return ser, cfg, np.arange(len(t)), np.arange(len(t))


def _sts_null(calib, t_cal, first_day, abnormal, seed):
    """No-change data from the patient's STS model, through the real pipeline.

    Fitted on the original-resolution readings strictly before `t_cal` — the first worn
    day at arming, the trailing `fit_days` afterwards, with abnormal days left out of the
    fit (their timing still shapes the mask). Simulated on that history's own
    observation mask, smoothed by the same smoother with the same settings, turned into
    the same column, binned, and whitened by the same noise model.
    -> (twin-like tuple, fit) or (None, fit-or-None) if it cannot."""
    o = calib["orig"]
    sts_cfg = calib["sts_cfg"]
    lo = (o["stamp"][0] if first_day else
          t_cal - np.timedelta64(int(sts_cfg["fit_days"] * 1440), "m"))
    h = (o["stamp"] >= lo) & (o["stamp"] < t_cal)
    hs, hv, hk = o["stamp"][h], o["raw"][h].copy(), o["state"][h]
    if abnormal:
        hv[np.isin(hs.astype("datetime64[D]"), list(abnormal))] = np.nan
    f = _sts.fit(hs, hv, hk, sts_cfg)
    if f is None:
        return None, None
    cfg0 = calib["cfg0"]                       # unbinned: the synthetic is binned below
    need = cfg0["null_days"] * 1440 + cfg0["burn_in_min"]
    ss, sv, sk = _sts.simulate(f, hs, hk, need, sts_cfg, seed=seed)
    if sv.size < 10:
        return None, f
    frame = pd.DataFrame({"createdTime": pd.to_datetime(ss), calib["val"]: sv})
    frame = _ensure_chunk_id(frame)
    if calib["smoother"] is not None:
        name, over = calib["smoother"]
        spec = registry.get("smoothing", name)
        sm = spec.entry()(frame.assign(patient_id="sts"), patient_col="patient_id",
                          timestamp_col="createdTime", value_col=calib["val"],
                          config=spec.config(**over), out_col="_sm")
        sm = sm.sort_values("createdTime").reset_index(drop=True)
        col = (sm[calib["val"]] - sm["_sm"] if calib["on"] == "residual" else sm["_sm"])
        frame = frame.assign(_col=col.to_numpy())
    else:
        frame = frame.assign(_col=frame[calib["val"]])
    ok = np.flatnonzero(np.isfinite(frame["_col"].to_numpy(float)))
    if ok.size < 10:
        return None, f
    st = frame["createdTime"].to_numpy("datetime64[ns]")[ok]
    tt = (st - st[0]).astype("timedelta64[s]").astype(float) / 60.0
    ser, c2, _, _ = _prepare(st, tt, frame["_col"].to_numpy(float)[ok],
                             frame[calib["val"]].to_numpy(float)[ok],
                             frame["chunk_id"].to_numpy()[ok], sk[ok], cfg0)
    tw = _whiten.noise_model(ser["stamp"], ser["t"], ser["r"], ser["raw"], ser["chunk"],
                             c2, ser["state"])
    worn = tw["armed"].sum() * c2["step_min"] / 1440.0
    return (ser["t"], tw["z"], ser["chunk"], tw["armed"], worn, tw.get("nu", 200.0)), f


def _thresholds(modules, settings, nm, ser, cfg, budget, calib=None):
    """Per detector and statistic, a per-reading threshold: the value at which the
    ensemble fires `budget` times per worn day on no-change data for this patient.
    -> (thresholds, one report row per recalibration).

    The no-change data is rebuilt at arming from the first worn day, then every
    `recal_days` from trailing history, and each threshold holds until the next rebuild.
    It is either a twin of resampled whole days (`calibration="shuffle"`, the default,
    `_whiten.twin`) or a simulation from the patient's structural time-series model
    (`"sts"`, `_sts_null`), which falls back to the twin when the model cannot be fitted.
    Everything it uses is from before the point it applies from: causal. The budget is
    split evenly across detectors, then across a detector's statistics; a threshold
    counts complete alarm events, not readings over it."""
    method = (calib or {}).get("method", "shuffle")
    n, share = len(ser["t"]), budget / len(modules)
    out = {name: {} for name, _ in modules}
    for name, mod in modules:
        if hasattr(mod, "threshold_for"):                 # analytic (drift): scalar
            out[name] = mod.threshold_for(share, **settings[name])
    sim = [(name, mod) for name, mod in modules
           if hasattr(mod, "score") and not hasattr(mod, "threshold_for")]
    rows = []
    if not sim:
        return out, rows
    stamp, arm = ser["stamp"], nm["arm"]
    day = stamp.astype("datetime64[D]")
    first_of_day = np.r_[0, np.flatnonzero(day[1:] != day[:-1]) + 1]
    points, nxt = [arm], stamp[arm] + np.timedelta64(int(cfg["recal_days"] * 1440), "m")
    for i in first_of_day:
        if i > arm and stamp[i] >= nxt:
            points.append(int(i))
            nxt = stamp[i] + np.timedelta64(int(cfg["recal_days"] * 1440), "m")
    for k, p0 in enumerate(points):
        p1 = points[k + 1] if k + 1 < len(points) else n
        abnormal = set() if k == 0 else set(nm["abnormal"])
        null, fit, used = None, None, "shuffle"
        if method == "sts":
            null, fit = _sts_null(calib, stamp[p0], k == 0, abnormal, seed=k)
            used = ("sts_" + fit["model"]) if null is not None else "shuffle_fallback"
        if null is None:
            if k == 0:
                days = np.unique(day[:arm])                 # the first worn day
            else:
                lo = stamp[p0] - np.timedelta64(int(cfg["baseline_days"] * 1440), "m")
                days = np.unique(day[(stamp >= lo) & (np.arange(n) < p0)])
                days = [d for d in days if d not in abnormal] or list(np.unique(day[:p0]))
            null = _whiten.twin(stamp, ser["r"], ser["raw"], ser["chunk"], days, cfg,
                                seed=k, state=ser["state"])
        t, z, chunk, armed, worn, _ = null
        row = {"recal": k, "from": stamp[p0], "calibration": used,
               "null_worn_days": round(float(worn), 2), **_sts.report(fit)}
        pri = {"chunk": chunk, "step_min": cfg["step_min"], "nu": nm["nu"]}
        for name, mod in sim:
            streams = mod.score(t, z, pri, **settings[name])
            for sname, v in streams.items():
                th = _whiten.pick_threshold(v, armed, worn, share / len(streams))
                out[name].setdefault(sname, np.full(n, np.inf))[p0:p1] = th
                hit = np.nan_to_num(np.asarray(v, float)[armed], nan=-np.inf) >= th
                row[f"thr_{name}_{sname}"] = th
                row[f"null_rate_{name}_{sname}"] = (_whiten.events(hit) / worn
                                                    if worn > 0 else np.nan)
        for name, mod in modules:
            if hasattr(mod, "threshold_for"):
                for sname, th in out[name].items():
                    row[f"thr_{name}_{sname}"] = th
        rows.append(row)
    return out, rows


def _flags(pid: str):
    """The patient's minute sleep / steps flags from ingest, or None if either is absent."""
    out = []
    for ch in ("sleep_active", "steps_active"):
        f = processed_path(pid, ch)
        if not f.exists():
            return None
        d = pd.read_csv(f, usecols=["createdTime", ch], parse_dates=["createdTime"])
        out.append(d.dropna().set_index("createdTime")[ch])
    return out


def _detect_patient(df: pd.DataFrame, col: str, raw_col: str, method_names, settings,
                    cfg, budget, calib=None):
    """CPD for one patient on one column: (bin) -> noise model -> calibration ->
    detectors -> back onto the readings.

    Everything is causal. The noise model arms after the patient's first worn day and
    whitens the column into z; thresholds come from no-change twins of the patient's own
    past days; the detectors run once over the whole patient (no per-chunk restart: they
    handle elapsed time, and the noise model clears its memory at an H8 reset). With
    `bin_min` > 0 (HR) all of this runs on median bins, and each reading holds the
    output of the latest bin closed at or before it.

    `calib` selects the calibration (None = day shuffling). For STS it carries
    {"method": "sts", "sts_cfg", "smoother": (name, settings) or None, "on", "val"}.
    The per-recalibration report rows are returned in `out.attrs["calibration"]`."""
    df = df.sort_values("createdTime").reset_index(drop=True)
    df = _ensure_chunk_id(df)
    sig = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    obs = np.flatnonzero(np.isfinite(sig))
    if obs.size < 2:
        return None
    modules = [(m, registry.get("cpd", m).load()) for m in method_names]
    stamp = df["createdTime"].to_numpy("datetime64[ns]")[obs]
    t = (stamp - stamp[0]).astype("timedelta64[s]").astype(float) / 60.0
    raw = (pd.to_numeric(df[raw_col], errors="coerce").to_numpy(float)[obs]
           if raw_col in df else sig[obs])
    chunk = df["chunk_id"].to_numpy()[obs]
    # Behavioural state per reading (sleep / active / rest): the noise model compares a
    # reading with readings in the same state, on top of the hour-of-day profile.
    fl = _flags(str(df["p_id"].iloc[0]))
    state = (None if fl is None else
             _whiten.reading_state(stamp, fl[0], fl[1], cfg["state_window_min"]))
    if calib is not None and calib.get("method") == "sts":
        # STS needs the original-resolution readings and the unbinned settings, since
        # its synthetic data is binned exactly as the real data is
        calib = {**calib, "cfg0": cfg,
                 "orig": {"stamp": stamp, "raw": raw,
                          "state": state if state is not None else np.zeros(obs.size, int)}}
    ser, cfg, holder, last = _prepare(stamp, t, sig[obs], raw, chunk, state, cfg)
    nm = _whiten.noise_model(ser["stamp"], ser["t"], ser["r"], ser["raw"], ser["chunk"],
                             cfg, ser["state"])

    n, nb = len(df), len(ser["t"])
    per_b = {name: _blank(nb) for name, _ in modules}
    report = []
    if nm["arm"] is not None:
        th, report = _thresholds(modules, settings, nm, ser, cfg, budget, calib)
        base = {"chunk": ser["chunk"], "stamp": ser["stamp"], "raw": ser["raw"],
                "step_min": cfg["step_min"], "nu": nm["nu"], "armed": nm["armed"],
                "state": ser["state"]}
        for name, mod in modules:
            per_b[name] = mod.detect(ser["t"], nm["z"], {**base, "thresholds": th[name]},
                                     **settings[name])
    # back onto the readings: each holds its latest closed bin (or itself, unbinned)
    held = holder >= 0
    rows = obs[held]
    src = holder[held]
    armed = np.zeros(n, bool)
    armed[rows] = nm["armed"][src]
    per = {name: _blank(n) for name, _ in modules}
    for name, _ in modules:
        for k in ("anomaly", "phenotype", "magnitude", "severity"):
            per[name][k][rows] = np.asarray(per_b[name][k])[src]

    out = df[["p_id", "createdTime", col, "chunk_id"]].copy()
    out["armed"] = armed.astype(int)
    for name, _ in modules:
        a = per[name]["anomaly"] & armed            # suppressed during burn-in / settling
        out[f"{name}_anomaly"] = a.astype(int)
        ph = per[name]["phenotype"].copy()
        ph[~armed] = "burn_in"
        ph[(a == 0) & armed] = "normal"
        out[f"{name}_phenotype"] = ph
        out[f"{name}_magnitude"] = per[name]["magnitude"]

    # Ensemble: OR across methods (recall-first); the reported phenotype comes from the
    # firing method with the largest severity, ties going to the earliest method.
    out["anomaly_present"], out["primary_phenotype"], out["magnitude"] = _vote(
        [per[n_] for n_, _ in modules], armed)
    # the noise model's own outputs; join turns them into day features. z and scale are
    # held like the flags; a re-entry jump is written once, on its bin's last reading.
    for k, name, at in (("z", "noise_z", None), ("scale", "noise_scale", None),
                        ("reentry_jump_z", "reentry_jump_z", "last"),
                        ("reentry_gap_min", "reentry_gap_min", "last")):
        v = np.full(n, np.nan)
        if at is None:
            v[rows] = nm[k][src]
        else:
            v[obs[last]] = nm[k]
        out[name] = v
    out.attrs["calibration"] = [{"p_id": str(df["p_id"].iloc[0]), **r} for r in report]
    return out


def input_signal(input_artifact, signal: str | None = None) -> str:
    """The channel to read: the argument, else whatever the input artifact recorded."""
    if signal is not None:
        return signal
    if input_artifact is not None:
        return input_artifact.manifest.get("stats", {}).get("signal", DEFAULT_CHANNEL)
    return DEFAULT_CHANNEL


def _load_input(input_artifact, on: str, signal: str = DEFAULT_CHANNEL) -> pd.DataFrame:
    col = signal_column(on, signal)
    if on == "raw" and input_artifact is None:
        frames = []
        for f in processed_files(signal):
            d = pd.read_csv(f, encoding="utf-8-sig")
            d["p_id"] = processed_pid(f)
            frames.append(d)
        if not frames:
            raise FileNotFoundError(
                f"no {signal} files in {PROCESSED_DIR} — run `ingest` first")
        frame = pd.concat(frames, ignore_index=True)
    else:
        if input_artifact is None:
            kind = "forecast" if on in ("surprise", "tail") else "smooth"
            raise ValueError(f"on={on!r} needs a {kind} artifact as input")
        frame = input_artifact.read_table(input_artifact.manifest["tables"][0])
    if col not in frame.columns:
        raise ValueError(f"input has no column {col!r} for on={on!r}; "
                         f"has {list(frame.columns)}")
    frame["createdTime"] = pd.to_datetime(frame["createdTime"])
    return frame


def _patient_job(g, **kw):
    """Worker: one patient's table and its calibration report rows (pandas attrs do not
    reliably cross a process boundary, so the rows travel beside the table).

    One maths thread per worker: the pool already runs a worker per core, and letting
    each one's BLAS (statsmodels, numpy) spawn its own threads oversubscribed a 40-core
    machine to a load of ~350 and stalled the STS run."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        out = _detect_patient(g, **kw)
    return out, ([] if out is None else out.attrs.get("calibration", []))


def _calibration_context(calibration, on, input_artifact, signal, step_cfg):
    """What STS needs to push synthetic data through the same pipeline as the real data:
    the smoother and its settings (from the input artifact), the column kind, the
    channel's value column and the model settings. None for the default shuffle."""
    if calibration not in CPD_CALIBRATIONS:
        raise ValueError(f"calibration must be one of {CPD_CALIBRATIONS}, got {calibration!r}")
    if calibration == "shuffle":
        return None
    if on not in ("raw", "smoothed", "residual"):
        raise ValueError(
            f"calibration='sts' needs on='raw', 'smoothed' or 'residual', got {on!r}: a "
            f"forecast input would need the frozen forecaster run on every simulation")
    smoother = None
    if on != "raw":
        name = input_artifact.method.split("@")[0]
        spec = registry.get("smoothing", name)
        smoother = (name, {k: v for k, v in input_artifact.manifest["params"].items()
                           if k in spec.defaults})
    return {"method": "sts", "on": on, "val": CHANNELS[signal], "smoother": smoother,
            "sts_cfg": {**_sts.DEFAULTS, **STS_SETTINGS.get(signal, {}),
                        "reading_min": step_cfg["step_min"]}}


def run(methods=cpd.DEFAULT_ENSEMBLE, on: str = "residual", input_artifact=None,
        force: bool = False, limit: int | None = None, signal: str | None = None,
        budget: float = CPD_BUDGET, calibration: str = CPD_CALIBRATION):
    """`signal` picks the channel; omitted, it is taken from the input artifact's own
    manifest, so a smooth artifact built with `--signal hr` is detected on heart rate
    without having to say so twice. `budget` is how often the ensemble may fire per worn
    day on no-change noise built from each patient's own data (config.CPD_BUDGETS).
    `calibration` is how that no-change data is built: "shuffle" (default) or "sts"
    (config.CPD_CALIBRATIONS); both write a `calibration` table beside `detected`."""
    if on not in SIGNALS:
        raise ValueError(f"on must be one of {sorted(SIGNALS)}, got {on!r}")
    signal = input_signal(input_artifact, signal)
    methods = tuple(methods)
    specs = [registry.get("cpd", m) for m in methods]
    col = signal_column(on, signal)

    # The artifact's "method" is the ensemble it ran, so variants sit side by side.
    # A non-default channel is suffixed as in smooth/forecast, so an HR and an HRV run
    # of the same ensemble never share a name — or a column prefix in `join`.
    name = "+".join(methods) + ("" if signal == DEFAULT_CHANNEL else f"@{signal}")
    step_knobs, settings = cpd_settings(signal, methods)
    params = {"on": on, "signal_col": col, "methods": list(methods),
              "detect": step_knobs, "settings": settings, "budget": float(budget)}
    calib = _calibration_context(calibration, on, input_artifact, signal, step_knobs)
    if calib is not None:                    # keyed only when not the default
        params = {**params, "calibration": calibration, "sts": calib["sts_cfg"]}
    if limit:
        params["limit_patients"] = int(limit)
    inputs = [input_artifact.ref] if input_artifact is not None else [
        {"step": "ingest", "method": "processed_dir", "key": str(PROCESSED_DIR)}]
    # One code hash for the whole ensemble: edit any detector and this recomputes.
    code_hash = "-".join(s.source_hash[:8] for s in specs)

    if not force:
        hit = store.find(STEP, name, params, inputs, code_hash)
        if hit is not None:
            print(f"[detect/{name} on {on}] reusing {hit.key[:12]}")
            return hit

    frame = _load_input(input_artifact, on, signal=signal)
    if limit:
        frame = frame[frame["p_id"].isin(sorted(frame["p_id"].unique())[:int(limit)])]
    art = store.open_artifact(STEP, name, params, inputs, code_hash,
                              label=f"{name} on {on}")
    print(f"\n[detect/{name}] on {on} ({col})")
    print(f"  input   {frame['p_id'].nunique()} patients, {len(frame):,} rows")
    print(f"  key     {art.key[:12]}   budget {budget:.3g}/worn day")

    # Patients are independent, so they run in parallel; map keeps their order, and
    # every detector seeds its own RNG per call, so the output is byte-reproducible.
    groups = [g for _, g in frame.groupby("p_id", sort=True)]
    work = partial(_patient_job, col=col, raw_col=CHANNELS[signal],
                   method_names=methods, settings=settings, cfg=step_knobs,
                   budget=float(budget), calib=calib)
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        done = list(tqdm(ex.map(work, groups), total=len(groups), desc=f"detect {on}"))
    outs = [o for o, _ in done if o is not None]
    calib_rows = [r for _, rows in done for r in rows]

    if not outs:
        art.fail("no patient had enough valid samples")
        raise RuntimeError(f"detect/{name}: nothing to annotate")

    table = pd.concat(outs, ignore_index=True)
    art.write_table("detected", table)         # first: `join` reads tables[0]
    art.write_table("calibration", pd.DataFrame(calib_rows))

    armed_rows = table[table["armed"] == 1]

    def _events(c):
        """Runs of a flag over each patient's armed readings (a settle window between
        two flagged readings does not make them two events)."""
        return int(sum(_whiten.events(g.to_numpy() == 1)
                       for _, g in armed_rows.groupby("p_id", sort=False)[c]))

    # Compare methods by EVENTS (contiguous runs), not flagged rows: a sustained-regime
    # detector like the HMM marks every sample of a regime.
    ev = {f"{m}_events": _events(f"{m}_anomaly") for m in methods}
    used = pd.Series([r["calibration"] for r in calib_rows]).value_counts().to_dict()
    art.complete(budget=float(budget), calibration=calibration, calibration_used=used,
                 n_patients=len(outs), n_rows=len(table),
                 n_armed=int(table["armed"].sum()),
                 n_anomaly=int(table["anomaly_present"].sum()),
                 ensemble_events=_events("anomaly_present"), **ev)
    print(f"  -> {len(outs)} patients, {ev}, ensemble "
          f"{art.manifest['stats']['ensemble_events']} events ({art.dir})")
    return art
