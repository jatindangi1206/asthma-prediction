"""Step: change-point detection. Runs CPD methods on whichever signal you point it at.

    input   `on="raw"`                 -> the ingest artifact (data/processed/)
            `on="smoothed"|"residual"` -> a smooth artifact
    output  one long table `detected`:
            p_id, createdTime, <signal>, chunk_id, armed,
            <method>_anomaly / _phenotype / _magnitude  for each method run,
            anomaly_present, primary_phenotype, magnitude

Some methods work better on a raw signal and some on a de-trended one, so the input
column is an argument, not an architecture. `residual` keeps the dual-track rationale
(detection on raw false-alarms on circadian drift; on smoothed it loses the volatility
bursts) while remaining one option among three.

Depends on no other step: it is handed an artifact and a column name.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from wearable import cpd, registry, store
from wearable.config import BURN_IN_N, GAP_THRESHOLD_MIN, POP_N0, POP_SIGMA0_DEFAULT, PROCESSED_DIR

warnings.filterwarnings("ignore")

STEP = "detect"
SIGNALS = {"raw": "hrvValue", "smoothed": "smoothed_hrv", "residual": "residual_hrv"}


def _population_sigma0(frame: pd.DataFrame, col: str) -> float:
    """Median per-patient robust std (1.4826·MAD) across the cohort — the prior each
    patient's own burn-in then shrinks toward."""
    sigmas = []
    for _, g in frame.groupby("p_id", sort=False):
        v = pd.to_numeric(g[col], errors="coerce").to_numpy()
        v = v[np.isfinite(v)]
        if v.size >= 30:
            sigmas.append(1.4826 * np.median(np.abs(v - np.median(v))))
    return float(np.median(sigmas)) if sigmas else POP_SIGMA0_DEFAULT


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


def _detect_patient(df: pd.DataFrame, col: str, methods, pop_sigma0: float):
    """CPD for one patient on one column. Per-chunk reset (H8), cumulative burn-in."""
    df = df.sort_values("createdTime").reset_index(drop=True)
    df = _ensure_chunk_id(df)

    sig = pd.to_numeric(df[col], errors="coerce").to_numpy()
    observed = np.isfinite(sig)
    obs_idx = np.where(observed)[0]
    if obs_idx.size < 2:
        return None

    elapsed = ((df["createdTime"] - df["createdTime"].iloc[0])
               .dt.total_seconds().to_numpy() / 60.0)

    # Cold start: the first BURN_IN_N VALID samples (counted across chunks, ignoring
    # MNAR gaps) set the noise floor, shrunk toward the population prior.
    burn = sig[obs_idx[:BURN_IN_N]]
    n = int(burn.size)
    m_obs = float(np.median(burn))
    sig_obs = float(max(1.4826 * np.median(np.abs(burn - m_obs)), np.std(burn), 1e-3))
    sigma0 = float(np.sqrt((POP_N0 * pop_sigma0 ** 2 + n * sig_obs ** 2) / (POP_N0 + n)))
    m0 = float(n * m_obs / (POP_N0 + n))
    priors = {"m0": m0, "sigma0": sigma0}

    armed = np.zeros(len(df), bool)
    if obs_idx.size > BURN_IN_N:
        armed[obs_idx[BURN_IN_N:]] = True

    per = {name: {"anomaly": np.zeros(len(df), int),
                  "phenotype": np.full(len(df), "normal", dtype=object),
                  "magnitude": np.zeros(len(df)),
                  "severity": np.zeros(len(df))} for name, _ in methods}

    for cid in df["chunk_id"].unique():
        idx = np.where(observed & (df["chunk_id"].to_numpy() == cid))[0]
        if idx.size < 2:
            continue
        t = elapsed[idx] - elapsed[idx[0]]
        v = sig[idx]
        for name, module in methods:
            r = module.detect(t, v, priors)
            for k in ("anomaly", "phenotype", "magnitude", "severity"):
                per[name][k][idx] = r[k]

    out = df[["p_id", "createdTime", col, "chunk_id"]].copy()
    out["armed"] = armed.astype(int)
    for name, _ in methods:
        a = per[name]["anomaly"] & armed            # suppressed during burn-in
        out[f"{name}_anomaly"] = a.astype(int)
        ph = per[name]["phenotype"].copy()
        ph[~armed] = "burn_in"
        ph[(a == 0) & armed] = "normal"
        out[f"{name}_phenotype"] = ph
        out[f"{name}_magnitude"] = per[name]["magnitude"]

    # Ensemble: OR across methods (recall-first); the reported phenotype comes from the
    # firing method with the largest severity, ties going to the earliest method.
    out["anomaly_present"], out["primary_phenotype"], out["magnitude"] = _vote(
        [per[n] for n, _ in methods], armed)
    out["sigma0"] = sigma0
    return out


def _load_input(input_artifact, on: str) -> pd.DataFrame:
    col = SIGNALS[on]
    if on == "raw" and input_artifact is None:
        frames = []
        for f in sorted(PROCESSED_DIR.glob("*_processed.csv")):
            d = pd.read_csv(f, encoding="utf-8-sig")
            d["p_id"] = f.stem.replace("_processed", "")
            frames.append(d)
        if not frames:
            raise FileNotFoundError(f"no processed CSVs in {PROCESSED_DIR}")
        frame = pd.concat(frames, ignore_index=True)
    else:
        if input_artifact is None:
            raise ValueError(f"on={on!r} needs a smooth artifact as input")
        frame = input_artifact.read_table("smoothed")
    if col not in frame.columns:
        raise ValueError(f"input has no column {col!r} for on={on!r}; "
                         f"has {list(frame.columns)}")
    frame["createdTime"] = pd.to_datetime(frame["createdTime"])
    return frame


def run(methods=cpd.DEFAULT_ENSEMBLE, on: str = "residual", input_artifact=None,
        force: bool = False, limit: int | None = None):
    if on not in SIGNALS:
        raise ValueError(f"on must be one of {sorted(SIGNALS)}, got {on!r}")
    methods = tuple(methods)
    specs = [registry.get("cpd", m) for m in methods]
    col = SIGNALS[on]

    # The artifact's "method" is the ensemble it ran, so variants sit side by side.
    name = "+".join(methods)
    params = {"on": on, "signal_col": col, "methods": list(methods),
              "burn_in_n": BURN_IN_N, "pop_n0": POP_N0}
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

    frame = _load_input(input_artifact, on)
    if limit:
        frame = frame[frame["p_id"].isin(sorted(frame["p_id"].unique())[:int(limit)])]
    loaded = [(m, registry.get("cpd", m).load()) for m in methods]
    art = store.open_artifact(STEP, name, params, inputs, code_hash,
                              label=f"{name} on {on}")
    pop = _population_sigma0(frame, col)
    print(f"\n[detect/{name}] on {on} ({col})")
    print(f"  input   {frame['p_id'].nunique()} patients, {len(frame):,} rows")
    print(f"  key     {art.key[:12]}   pop prior sigma0={pop:.2f}")

    outs = []
    for _, g in tqdm(frame.groupby("p_id", sort=True), desc=f"detect {on}"):
        r = _detect_patient(g, col, loaded, pop)
        if r is not None:
            outs.append(r)

    if not outs:
        art.fail("no patient had enough valid samples")
        raise RuntimeError(f"detect/{name}: nothing to annotate")

    table = pd.concat(outs, ignore_index=True)
    art.write_table("detected", table)

    def _events(a):
        a = np.asarray(a).astype(int)
        return int(((a == 1) & (np.r_[0, a[:-1]] == 0)).sum())

    # Compare methods by EVENTS (contiguous runs), not flagged rows: a sustained-regime
    # detector like the HMM marks every sample of a regime.
    ev = {f"{m}_events": _events(table[f"{m}_anomaly"]) for m in methods}
    art.complete(n_patients=len(outs), n_rows=len(table),
                 n_armed=int(table["armed"].sum()),
                 n_anomaly=int(table["anomaly_present"].sum()),
                 ensemble_events=_events(table["anomaly_present"]), **ev)
    print(f"  -> {len(outs)} patients, {ev}, ensemble "
          f"{art.manifest['stats']['ensemble_events']} events ({art.dir})")
    return art
