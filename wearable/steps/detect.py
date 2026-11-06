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
detector's threshold on no-change noise built from past history. This is an initial
candidate fix; prospective validity also requires a causal upstream transform.

Some methods work better on a raw signal and some on a de-trended one, so the input
column is an argument, not an architecture. `residual` keeps the dual-track rationale
(detection on raw false-alarms on circadian drift; on smoothed it loses the volatility
bursts) while remaining one option among three.

Depends on no other step: it is handed an artifact and a column name.
"""
from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path
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
from wearable.cpd._contracts import positive, starts
from wearable.chunking import assign_chunks

STEP = "detect"
LOGIC = "cpd_initial_candidate_1"
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


def _ensure_chunk_id(df: pd.DataFrame, value_col=None) -> pd.DataFrame:
    """Canonical H8 segmentation on real observations, never filler adjacency."""
    df = df.copy()
    value_col = value_col or next(c for c in CHANNELS.values() if c in df.columns)
    finite = np.isfinite(pd.to_numeric(df[value_col], errors="coerce"))
    obs = df.loc[finite, ["createdTime"]].copy()
    obs["_row"] = np.flatnonzero(finite)
    obs["_patient"] = "one"
    obs = assign_chunks(obs, "createdTime", "_patient")
    chunks = np.full(len(df), np.nan)
    chunks[obs["_row"].to_numpy(int)] = obs["chunk_id"].to_numpy()
    df["chunk_id"] = chunks
    return df


def _vote(per: list[dict], armed: np.ndarray):
    """OR eligible hits; representative phenotype follows configured method order.

    Unlike posterior probabilities and threshold-relative scores are not compared.
    """
    anom = np.stack([p["anomaly"] for p in per]) == 1
    fired = anom & armed
    claims = fired  # representative phenotype follows declared ensemble order
    has = claims.any(0)
    best = claims.argmax(0)

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
        ser = _whiten.bins(stamp, t, r, raw, chunk, cfg["bin_min"], state,
                           reading_min=cfg["step_min"])
        return (ser, {**cfg, "step_min": max(cfg["step_min"], cfg["bin_min"])},
                ser["holder"], ser["last"])
    ser = {"stamp": stamp, "t": t, "r": r, "raw": raw, "chunk": chunk, "state": state,
           "exposure": np.full(len(t), cfg["step_min"])}
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
                             c2, ser["state"], ser["exposure"])
    worn = float(ser["exposure"][tw["armed"]].sum()) / 1440.0
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
    Histories are timestamp-bounded before activation. The budget is
    split evenly across detectors, then across a detector's statistics; a threshold
    counts complete alarm events, not readings over it."""
    method = (calib or {}).get("method", "shuffle")
    n, share = len(ser["t"]), budget / len(modules)
    out = {name: {s: np.full(n, np.inf) for s in mod.THRESHOLDS} for name, mod in modules}
    for name, mod in modules:
        if hasattr(mod, "threshold_for"):                 # analytic (drift): scalar
            out[name] = mod.threshold_for(share, **settings[name])
    sim = [(name, mod) for name, mod in modules
           if hasattr(mod, "score") and not hasattr(mod, "threshold_for")]
    rows = []
    stamp, arm = ser["stamp"], nm["arm"]
    day = stamp.astype("datetime64[D]")
    first_of_day = np.r_[0, np.flatnonzero(day[1:] != day[:-1]) + 1]
    points, nxt = list(nm.get("arms", [arm])), stamp[arm] + np.timedelta64(int(cfg["recal_days"] * 1440), "m")
    for i in first_of_day:
        if i > arm and stamp[i] >= nxt:
            points.append(int(i))
            nxt = stamp[i] + np.timedelta64(int(cfg["recal_days"] * 1440), "m")
    points = sorted(set(points))
    for k, p0 in enumerate(points):
        p1 = points[k + 1] if k + 1 < len(points) else n
        # Only completed calendar days may influence this as-of decision.
        abnormal = {d for d in nm["abnormal"] if d < day[p0]}
        null, fit, used = None, None, "shuffle"
        failure = ""
        if method == "sts":
            try:
                null, fit = _sts_null(calib, stamp[p0], k == 0, abnormal, seed=k)
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"
            used = ("sts_" + fit["model"]) if null is not None else "shuffle_fallback"
        if null is None:
            if k == 0:
                days = np.unique(day[:arm])                 # the first worn day
            else:
                lo = stamp[p0] - np.timedelta64(int(cfg["baseline_days"] * 1440), "m")
                days = np.unique(day[(stamp >= lo) & (np.arange(n) < p0)])
                days = [d for d in days if d not in abnormal]
            lower = (stamp[0] if k == 0 else
                     stamp[p0] - np.timedelta64(int(cfg["baseline_days"] * 1440), "m"))
            try:
                null = _whiten.twin(stamp, ser["r"], ser["raw"], ser["chunk"], days, cfg,
                                    seed=k, state=ser["state"], cutoff=stamp[p0], lower=lower,
                                    exposure=ser["exposure"])
            except Exception as exc:
                failure += f" shuffle {type(exc).__name__}: {exc}"
        if null is None or null[4] <= 0:
            rows.append({"recal": k, "from": stamp[p0], "calibration": "disabled",
                         "reason": failure or "no eligible null exposure"})
            continue
        t, z, chunk, armed, worn, _ = null
        row = {"recal": k, "from": stamp[p0], "calibration": used,
               "null_worn_days": round(float(worn), 2), "fallback_reason": failure,
               "occupancy_cap": cfg["max_null_occupancy"], **_sts.report(fit)}
        pri = {"chunk": chunk, "step_min": cfg["step_min"],
               "nu": float(nm["tail_nu"][p0]) if np.isfinite(nm["tail_nu"][p0]) else 200.0,
               "tail_scale": float(nm["tail_scale"][p0]) if np.isfinite(nm["tail_scale"][p0]) else 1.0}
        combined = np.zeros(len(t), bool)
        for name, mod in sim:
            streams = mod.score(t, z, pri, **settings[name])
            for sname, v in streams.items():
                th = _whiten.pick_threshold(v, armed, worn, share / len(streams),
                                            max_occupancy=cfg["max_null_occupancy"], time=t, chunk=chunk)
                out[name].setdefault(sname, np.full(n, np.inf))[p0:p1] = th
                hit = np.isfinite(v) & (np.asarray(v) >= th) & armed
                combined |= hit
                row[f"thr_{name}_{sname}"] = th
                row[f"null_occupancy_{name}_{sname}"] = float(hit.sum() / max(armed.sum(), 1))
                row[f"status_{name}_{sname}"] = "calibrated" if np.isfinite(th) else "disabled_no_scores"
                row[f"null_rate_{name}_{sname}"] = (starts(hit, t, chunk, GAP_THRESHOLD_MIN).sum() / worn
                                                    if worn > 0 else np.nan)
        for name, mod in modules:
            if hasattr(mod, "threshold_for"):
                for sname, th in out[name].items():
                    row[f"thr_{name}_{sname}"] = th
                    row[f"status_{name}_{sname}"] = "analytic_unvalidated"
        row["null_union_rate"] = starts(combined, t, chunk, GAP_THRESHOLD_MIN).sum() / worn
        row["null_union_occupancy"] = float(combined.sum() / max(armed.sum(), 1))
        rows.append(row)
    return out, rows


def _flags(pid: str):
    """Independent state sources: absence of one does not discard the other."""
    out = []
    for ch in ("sleep_active", "steps_active"):
        f = processed_path(pid, ch)
        if not f.exists():
            out.append(pd.Series(dtype=float, index=pd.DatetimeIndex([])))
            continue
        d = pd.read_csv(f, usecols=["createdTime", ch], parse_dates=["createdTime"])
        out.append(d.dropna().set_index("createdTime")[ch])
    return out


def _detect_patient(df: pd.DataFrame, col: str, raw_col: str, method_names, settings,
                    cfg, budget, calib=None):
    """Preserve raw support; infer on eligible transformed observations only.

    Closed-bin statistics publish forward on the original grid. Nightly drift reads
    the complete raw stream independently, not the smoother/forecaster's NaN mask.
    """
    df = df.copy()
    cfg0 = dict(cfg)
    df["createdTime"] = pd.to_datetime(df["createdTime"], errors="raise")
    df = df.sort_values("createdTime").reset_index(drop=True)
    if df["createdTime"].isna().any() or df["createdTime"].duplicated().any():
        raise ValueError("patient timestamps must be valid and unique")
    if raw_col not in df or col not in df:
        raise ValueError("CPD needs both the raw channel and selected input column")
    df = _ensure_chunk_id(df, raw_col)
    n = len(df)
    pid = str(df["p_id"].iloc[0])
    df["p_id"] = pid
    raw_values = pd.to_numeric(df[raw_col], errors="coerce").to_numpy(float)
    sig = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    raw_idx = np.flatnonzero(np.isfinite(raw_values))
    obs = np.flatnonzero(np.isfinite(sig) & np.isfinite(raw_values))
    modules = [(m, registry.get("cpd", m).load()) for m in method_names]
    fl = _flags(pid)
    raw_stamp = df["createdTime"].to_numpy("datetime64[ns]")[raw_idx]
    raw_state = _whiten.reading_state(raw_stamp, fl[0], fl[1], cfg["state_window_min"])
    out = df[list(dict.fromkeys(["p_id", "createdTime", raw_col, col, "chunk_id"]))].copy()
    per = {name: _blank(n) for name, _ in modules}
    armed = np.zeros(n, bool)
    report, fits = [], []
    for name in ("noise_z", "noise_scale", "reentry_jump_z", "reentry_gap_min"):
        out[name] = np.nan
    if len(obs) >= 2:
        stamp = df["createdTime"].to_numpy("datetime64[ns]")[obs]
        t = (stamp - stamp[0]).astype("timedelta64[s]").astype(float) / 60.0
        raw, chunk = raw_values[obs], df["chunk_id"].to_numpy()[obs]
        state = _whiten.reading_state(stamp, fl[0], fl[1], cfg["state_window_min"])
        cfg0 = dict(cfg)
        if calib is not None and calib.get("method") == "sts":
            calib = {**calib, "cfg0": cfg0,
                     "orig": {"stamp": raw_stamp, "raw": raw_values[raw_idx], "state": raw_state}}
        ser, cfg, holder, publish = _prepare(stamp, t, sig[obs], raw, chunk, state, cfg)
        nm = _whiten.noise_model(ser["stamp"], ser["t"], ser["r"], ser["raw"], ser["chunk"],
                                 cfg, ser["state"], ser["exposure"])
        fits = [{"p_id": pid, **f} for f in nm["fits"]]
        if nm["arm"] is not None:
            thresholds, report = _thresholds(modules, settings, nm, ser, cfg, budget, calib)
            base = {"chunk": ser["chunk"], "stamp": ser["stamp"], "raw": ser["raw"],
                    "step_min": cfg["step_min"], "armed": nm["armed"], "state": ser["state"],
                    "nu": np.where(np.isfinite(nm["tail_nu"]), nm["tail_nu"], 200.0),
                    "tail_scale": np.where(np.isfinite(nm["tail_scale"]), nm["tail_scale"], 1.0)}
            held = np.flatnonzero(holder >= 0)
            source = holder[held]
            valid = ((chunk[held] == ser["chunk"][source]) &
                     (t[held] - ser["t"][source] <= cfg["step_min"]) &
                     ~_whiten.settle_mask(t, chunk, cfg0)[held])
            at, source = obs[held[valid]], source[valid]
            armed[at] = nm["armed"][source]
            for name, mod in modules:
                if name == "drift":
                    continue
                result = mod.detect(ser["t"], nm["z"], {**base, "thresholds": thresholds[name]},
                                    **settings[name])
                for key in ("anomaly", "phenotype", "magnitude", "severity"):
                    per[name][key][at] = np.asarray(result[key])[source]
                available = np.asarray(result.get("available", np.isfinite(nm["z"])))
                availability = np.zeros(n, int)
                availability[at] = (available[source] & nm["armed"][source]).astype(int)
                out[f"{name}_available"] = availability
                for stream, values in result.get("scores", {}).items():
                    diagnostic = np.full(n, np.nan)
                    diagnostic[at] = np.asarray(values)[source]
                    out[f"{name}_score_{stream}"] = diagnostic
                    th = np.broadcast_to(thresholds[name][stream], len(ser["t"]))
                    diagnostic = np.full(n, np.nan)
                    diagnostic[at] = th[source]
                    out[f"{name}_threshold_{stream}"] = diagnostic
            out.loc[at, "noise_z"] = nm["z"][source]
            out.loc[at, "noise_scale"] = nm["scale"][source]
            for key in ("reentry_jump_z", "reentry_gap_min"):
                valid_pub = (publish < len(obs)) & (chunk[publish] == ser["chunk"])
                out.loc[obs[publish[valid_pub]], key] = nm[key][valid_pub]
    # Drift is a raw-channel verdict, independent of transformed-output support.
    if "drift" in method_names and len(raw_idx) >= 2:
        mod = dict(modules)["drift"]
        rt = (raw_stamp - raw_stamp[0]).astype("timedelta64[s]").astype(float) / 60.0
        rc = df["chunk_id"].to_numpy()[raw_idx]
        epoch_start = np.maximum.accumulate(np.where(
            np.r_[True, np.diff(rt) >= cfg0["baseline_days"] * 1440], np.arange(len(rt)), 0))
        raw_armed = (np.arange(len(raw_idx)) - epoch_start) * cfg0["step_min"] >= cfg0["burn_in_min"]
        raw_armed &= ~_whiten.settle_mask(rt, rc, cfg0)
        th = mod.threshold_for(budget / len(modules), **settings["drift"])
        result = mod.detect(rt, raw_values[raw_idx], {"stamp": raw_stamp, "raw": raw_values[raw_idx],
                            "chunk": rc, "state": raw_state, "step_min": cfg0["step_min"],
                            "armed": raw_armed, "thresholds": th}, **settings["drift"])
        for key in ("anomaly", "phenotype", "magnitude", "severity"):
            per["drift"][key][raw_idx] = result[key]
        armed[raw_idx] |= raw_armed
        out["drift_available"] = 0
        out.loc[raw_idx, "drift_available"] = (raw_armed & np.isfinite(result["magnitude"])).astype(int)
        report.append({"p_id": pid, "calibration": "analytic_unvalidated", "from": raw_stamp[0],
                       "thr_drift_level": th["level"], "exposure_unit": "eligible_night"})
    out["armed"] = armed.astype(int)
    for name, _ in modules:
        if f"{name}_available" not in out:
            out[f"{name}_available"] = 0
        eligible = out[f"{name}_available"].to_numpy(bool)
        per[name]["anomaly"] = (per[name]["anomaly"].astype(bool) & eligible).astype(int)
        out[f"{name}_anomaly"] = np.where(eligible, per[name]["anomaly"], np.nan)
        ph = per[name]["phenotype"].copy()
        ph[~eligible] = "unavailable"
        out[f"{name}_phenotype"] = ph
        out[f"{name}_magnitude"] = np.where(eligible, per[name]["magnitude"], np.nan)
    out["anomaly_present"], out["primary_phenotype"], out["magnitude"] = _vote(
        [per[name] for name, _ in modules], armed)
    any_available = out[[f"{name}_available" for name in method_names]].any(axis=1)
    out.loc[~any_available, ["anomaly_present", "magnitude"]] = np.nan
    out.loc[~any_available, "primary_phenotype"] = "unavailable"
    # Explicit event onsets survive midnight and are never counted through fillers.
    observed = raw_idx
    if len(observed):
        tt = (raw_stamp - raw_stamp[0]).astype("timedelta64[s]").astype(float) / 60
        ch = df["chunk_id"].to_numpy()[observed]
        for flag in ["anomaly_present", *[f"{m}_anomaly" for m in method_names]]:
            onset = np.full(n, np.nan)
            active = out[flag].fillna(0).to_numpy()[observed] == 1
            emitted = starts(active, tt, ch, GAP_THRESHOLD_MIN)
            eligible = out[flag].notna().to_numpy()[observed]
            onset[observed[eligible]] = emitted[eligible]
            out[f"{flag}_event_start"] = onset
    out.attrs["calibration"] = [{"p_id": pid, **r} for r in report]
    out.attrs["fits"] = fits
    out.attrs["status"] = ("processed" if any_available.any() else
                           "uncalibratable" if armed.any() else "unarmable")
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
    frame["p_id"] = frame["p_id"].astype(str)
    if input_artifact is not None:
        # A transform's NaN/output mask must not remove raw day features or nights.
        raw = _load_input(None, "raw", signal)
        value = CHANNELS[signal]
        if value in frame:
            overlap = frame[["p_id", "createdTime", value]].merge(
                raw[["p_id", "createdTime", value]], on=["p_id", "createdTime"],
                suffixes=("_artifact", "_current"), validate="one_to_one")
            old = pd.to_numeric(overlap[f"{value}_artifact"], errors="coerce").to_numpy(float)
            current = pd.to_numeric(overlap[f"{value}_current"], errors="coerce").to_numpy(float)
            if np.any(np.isfinite(old) & np.isfinite(current) &
                      ~np.isclose(old, current, rtol=1e-12, atol=1e-12)):
                raise ValueError("upstream transform and current raw data disagree; rebuild the transform")
        frame = frame.drop(columns=[value], errors="ignore").merge(
            raw[["p_id", "createdTime", value]], on=["p_id", "createdTime"],
            how="outer", validate="one_to_one")
    return frame


def _patient_job(g, **kw):
    """Worker: one patient's table and its calibration report rows (pandas attrs do not
    reliably cross a process boundary, so the rows travel beside the table).

    One maths thread per worker: the pool already runs a worker per core, and letting
    each one's BLAS (statsmodels, numpy) spawn its own threads oversubscribed a 40-core
    machine to a load of ~350 and stalled the STS run."""
    from threadpoolctl import threadpool_limits
    try:
        with threadpool_limits(1):
            out = _detect_patient(g, **kw)
        return out, out.attrs.get("calibration", []), out.attrs.get("fits", []), {
            "p_id": str(g["p_id"].iloc[0]), "status": out.attrs["status"], "reason": ""}
    except Exception as exc:
        # Preserve day/raw support. A failure is not a normal/no-alarm observation.
        cols = list(dict.fromkeys(["p_id", "createdTime", kw["raw_col"], kw["col"]]))
        out = g.reindex(columns=cols).copy()
        out["armed"] = 0
        out["chunk_id"] = np.nan
        out["anomaly_present"] = np.nan
        out["magnitude"] = np.nan
        out["primary_phenotype"] = "failed"
        for name in kw["method_names"]:
            out[f"{name}_available"] = 0
            out[f"{name}_anomaly"] = np.nan
            out[f"{name}_phenotype"] = "failed"
            out[f"{name}_magnitude"] = np.nan
        return out, [], [], {"p_id": str(g["p_id"].iloc[0]), "status": "failed",
                            "reason": f"{type(exc).__name__}: {exc}"}


def _dependencies(signal):
    """Hash every local file whose contents detection can read, including absence."""
    files = sorted(set(processed_files(signal) + processed_files("sleep_active") +
                       processed_files("steps_active")))
    h = hashlib.sha256()
    for path in files:
        h.update(str(path).encode())
        h.update(path.read_bytes())
    return {"step": "ingest", "method": f"channel_and_states:{signal}", "key": h.hexdigest()}


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
        budget: float = CPD_BUDGET, calibration: str = CPD_CALIBRATION, workers=None):
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
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("select at least one unique CPD method")
    if signal not in CPD_SETTINGS:
        raise ValueError(f"CPD settings are only defined for {list(CPD_SETTINGS)}")
    positive(budget=budget)
    if limit is not None:
        positive(limit=limit)
        if int(limit) != limit:
            raise ValueError("limit must be an integer")
    if workers is not None:
        positive(workers=workers)
        if int(workers) != workers:
            raise ValueError("workers must be an integer")
    specs = [registry.get("cpd", m) for m in methods]
    col = signal_column(on, signal)

    # The artifact's "method" is the ensemble it ran, so variants sit side by side.
    # A non-default channel is suffixed as in smooth/forecast, so an HR and an HRV run
    # of the same ensemble never share a name — or a column prefix in `join`.
    name = "+".join(methods) + ("" if signal == DEFAULT_CHANNEL else f"@{signal}")
    step_knobs, settings = cpd_settings(signal, methods)
    for key in ("step_min", "burn_in_min", "baseline_days", "null_days", "recal_days",
                "settle_min", "settle_h8_min", "reentry_gap_min", "state_window_min"):
        positive(**{key: step_knobs[key]})
    if (not np.isfinite(step_knobs["bin_min"]) or not 0 <= step_knobs["bin_min"] < GAP_THRESHOLD_MIN
            or not 0 < step_knobs["abnormal_frac"] <= 1 or not 0 < step_knobs["max_null_occupancy"] <= 1):
        raise ValueError("invalid bin size or abnormal fraction")
    for spec in specs:
        mod = spec.load()
        if not hasattr(mod, "THRESHOLDS") or not (hasattr(mod, "score") or hasattr(mod, "threshold_for")):
            raise ValueError(f"{spec.name} does not implement the CPD calibration contract")
    params = {"on": on, "signal_col": col, "methods": list(methods),
              "detect": step_knobs, "settings": settings, "budget": float(budget),
              "logic": LOGIC, "calibration": calibration}
    calib = _calibration_context(calibration, on, input_artifact, signal, step_knobs)
    if calib is not None:                    # keyed only when not the default
        params = {**params, "calibration": calibration, "sts": calib["sts_cfg"]}
    if limit:
        params["limit_patients"] = int(limit)
    inputs = ([input_artifact.ref] if input_artifact is not None else []) + [_dependencies(signal)]
    # One code hash for the whole ensemble: edit any detector and this recomputes.
    root = Path(__file__).resolve().parents[1]
    code_files = [Path(__file__), root / "config.py", root / "chunking.py",
                  root / "cpd" / "_contracts.py", root / "cpd" / "_whiten.py", root / "cpd" / "_sts.py"]
    code_hash = hashlib.sha256(("".join(s.source_hash for s in specs)).encode() +
                               b"".join(p.read_bytes() for p in code_files)).hexdigest()

    if not force:
        hit = store.find(STEP, name, params, inputs, code_hash)
        if hit is not None:
            print(f"[detect/{name} on {on}] reusing {hit.key[:12]}")
            return hit

    frame = _load_input(input_artifact, on, signal=signal)
    if limit:
        frame = frame[frame["p_id"].isin(sorted(frame["p_id"].unique())[:int(limit)])]
    art = store.open_artifact(STEP, name, params, inputs, code_hash,
                              label=f"{name} on {on}", cpd_schema=1,
                              cpd_validation="initial_candidate_not_runtime_validated")
    art.manifest["input_causality"] = (
        "raw" if on == "raw" else
        "causal_smoother_candidate" if input_artifact is not None and input_artifact.step == "smooth" and input_artifact.method.split("@")[0] == "anrewma" else
        "not_certified_by_cpd")
    print(f"\n[detect/{name}] on {on} ({col})")
    print(f"  input   {frame['p_id'].nunique()} patients, {len(frame):,} rows")
    print(f"  key     {art.key[:12]}   budget {budget:.3g}/worn day")

    # Patients are independent, so they run in parallel; map keeps their order, and
    # RNGs are seeded per call; cross-platform numerical reproducibility is unverified.
    groups = [g for _, g in frame.groupby("p_id", sort=True)]
    work = partial(_patient_job, col=col, raw_col=CHANNELS[signal],
                   method_names=methods, settings=settings, cfg=step_knobs,
                   budget=float(budget), calib=calib)
    try:
        if workers == 1:
            done = [work(g) for g in tqdm(groups, desc=f"detect {on}")]
        else:
            with ProcessPoolExecutor(workers or max(1, (os.cpu_count() or 2) - 1)) as ex:
                done = list(tqdm(ex.map(work, groups), total=len(groups), desc=f"detect {on}"))
        return _finish(art, done, methods, signal, budget, calibration)
    except Exception as exc:
        art.fail(f"{type(exc).__name__}: {exc}")
        raise


def _finish(art, done, methods, signal, budget, calibration):
    outs = [o for o, _, _, _ in done]
    calib_rows = [r for _, rows, _, _ in done for r in rows]

    if not outs:
        art.fail("no patient had enough valid samples")
        raise RuntimeError("detect: nothing to annotate")

    table = pd.concat(outs, ignore_index=True)
    art.write_table("detected", table)         # first: `join` reads tables[0]
    art.write_table("calibration", pd.DataFrame(calib_rows).reindex(
        columns=list(dict.fromkeys(["p_id", "from", "calibration", *[k for r in calib_rows for k in r]]))))
    art.write_table("noise_fits", pd.DataFrame([
        {k: json.dumps(v) if isinstance(v, list) else v for k, v in f.items()}
        for _, _, fits, _ in done for f in fits]).reindex(
            columns=["p_id", "from", "status", "n_history", "hour_center", "hour_scale",
                     "state_offset", "state_ratio", "ou_a", "ou_tau", "tail_nu", "tail_scale"]))
    statuses = pd.DataFrame([s for _, _, _, s in done])
    art.write_table("patient_status", statuses)
    failed = statuses[statuses["status"] == "failed"]
    if len(failed):
        art.write_table("failures", failed)

    def _events(c):
        """Runs of a flag over each patient's armed readings (a settle window between
        two flagged readings does not make them two events)."""
        return int(table.get(f"{c}_event_start", pd.Series(dtype=float)).sum())

    # Compare methods by EVENTS (contiguous runs), not flagged rows: a sustained-regime
    # detector like the HMM marks every sample of a regime.
    ev = {f"{m}_events": _events(f"{m}_anomaly") for m in methods}
    used = pd.Series([r["calibration"] for r in calib_rows], dtype=str).value_counts().to_dict()
    art.complete(signal=signal, budget=float(budget), calibration=calibration, calibration_used=used,
                 n_failed=len(failed), completeness="partial" if len(failed) else "complete",
                 exposure_definition="nominal_observed_support_minutes",
                 n_patients=len(outs), n_rows=len(table),
                 n_armed=int(table["armed"].sum()),
                 n_anomaly=int(table["anomaly_present"].sum()),
                 ensemble_events=_events("anomaly_present"), **ev)
    print(f"  -> {len(outs)} patients ({len(failed)} failed), {ev}, ensemble "
          f"{art.manifest['stats']['ensemble_events']} events ({art.dir})")
    return art
