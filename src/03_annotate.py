"""Stage 03 — CPD annotation on the RESIDUAL (Dual-Track Residual Architecture).

The four online CPD methods (BOCPD, Kernel-MMD, Kalman innovations, HMM) all run
on `residual_hrv = raw − smoothed`, NOT on raw or smoothed HRV. Each emits a Rich
Annotation Vector per timestep — {anomaly, primary_phenotype, magnitude} — and the
ensemble combines them into the ML target Y.

Pipeline rules baked in here:
  * Input column            : residual_hrv (asserted present).
  * 24-hour burn-in         : the first 144 observed residual points of each
                              PATIENT set the noise priors (median m0, std σ0) fed
                              to BOCPD (β0), Kalman (R) and the HMM (emission var).
                              Alarms are SUPPRESSED during this burn-in; afterward
                              the ensemble is "armed".
  * Per-chunk reset (H8)    : every detector runs independently per chunk_id; no
                              detector scans across a ≥180-min gap.
  * Ensemble vote           : anomaly_present = ANY of the four methods fires
                              (after arming); primary_phenotype = the firing method
                              with the highest normalized severity; magnitude = that
                              method's native magnitude.

Usage (from asthma-prediction/):
    python src/03_annotate.py [method]      # method default: gammadglm
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import bocpd
import kcp
import kalman_cpd
import hmm_cpd

METHODS = [("bocpd", bocpd), ("kcpd", kcp), ("kalman", kalman_cpd), ("hmm", hmm_cpd)]

# Track-A smoother -> (smoothed dir, file suffix). Pick which residual to annotate.
SMOOTHERS = {
    "pf":        ("data/smoothed",            "_smoothed"),   # NOTE: FFBS residual is non-causal
    "rspf":      ("data/smoothed_rspf",       "_rspf"),
    "krlst":     ("data/smoothed_krlst",      "_krlst"),
    "gpssm":     ("data/smoothed_gpssm",      "_gpssm"),
    "ossa":      ("data/smoothed_ossa",       "_ossa"),
    "kim":       ("data/smoothed_kim",        "_kim"),
    "gammadglm": ("data/smoothed_gammadglm",  "_gammadglm"),
    "logmhw":    ("data/smoothed_logmhw",     "_logmhw"),
    "hrkf":      ("data/smoothed_hrkf",       "_hrkf"),
    "anrewma":   ("data/smoothed_anrewma",    "_anrewma"),
    "ctimm":     ("data/smoothed_ctimm",      "_ctimm"),
}

GAP_THRESHOLD_MIN = 180.0
BURN_IN_N = 144              # 24h of valid samples to ARM the ensemble (counted cumulatively
                            #   across chunks, ignoring MNAR gaps)
POP_N0 = 30                 # pseudo-count of the population prior (Bayesian shrinkage strength)
POP_SIGMA0_DEFAULT = 15.0   # fallback population residual-noise prior (ms) if cohort unavailable
RESULTS_DIR = Path("./data/results")


def _population_sigma0(csv_files):
    """Population prior for the residual noise floor = median per-patient robust std
    (1.4826·MAD of residual_hrv) across the cohort. Robust starting point that the
    per-patient burn-in then shrinks toward (per-patient data dominates as it accrues)."""
    sigmas = []
    for f in csv_files:
        try:
            r = pd.to_numeric(pd.read_csv(f, usecols=["residual_hrv"])["residual_hrv"],
                              errors="coerce").to_numpy()
            r = r[np.isfinite(r)]
            if r.size >= 30:
                sigmas.append(1.4826 * np.median(np.abs(r - np.median(r))))
        except Exception:
            continue
    return float(np.median(sigmas)) if sigmas else POP_SIGMA0_DEFAULT


def _ensure_chunk_id(df):
    """Derive chunk_id from ≥180-min gaps if the smoother didn't emit one (e.g. PF)."""
    if "chunk_id" in df.columns:
        return df
    dt = df["createdTime"].diff().dt.total_seconds() / 60.0
    df["chunk_id"] = (dt.isna() | (dt >= GAP_THRESHOLD_MIN)).cumsum().astype(int)
    return df


def _load_calibration(file_path, calib_dir):
    """Stage-03a per-patient calibration, aligned to this file's grid by createdTime.
    Returns None when 03a hasn't been run — caller then uses the inline burn-in."""
    if calib_dir is None:
        return None
    cf = Path(calib_dir) / (file_path.stem + "_calib.csv")
    if not cf.exists():
        return None
    c = pd.read_csv(cf)
    c["createdTime"] = pd.to_datetime(c["createdTime"])
    return c


def annotate_patient(file_path, out_dir, pop_sigma0=None, calib_dir=None):
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        if "residual_hrv" not in df.columns:
            return {"file": file_path.name, "status": "failed",
                    "reason": "no residual_hrv column — re-run the smoother first"}

        df["createdTime"] = pd.to_datetime(df["createdTime"])
        df = df.sort_values("createdTime").reset_index(drop=True)
        df = _ensure_chunk_id(df)

        resid = pd.to_numeric(df["residual_hrv"], errors="coerce").to_numpy()
        observed = np.isfinite(resid)
        obs_idx = np.where(observed)[0]
        if obs_idx.size < 2:
            return {"file": file_path.name, "status": "failed", "reason": "no residuals"}

        elapsed = (df["createdTime"] - df["createdTime"].iloc[0]).dt.total_seconds().to_numpy() / 60.0

        calib = _load_calibration(file_path, calib_dir)
        if calib is not None:
            # ---- Track A': adaptive baseline + gap-aware arming from Stage 03a ----
            # 03a emits exactly one calib row per smoothed row, from the same file with the
            # same (stable) sort, so rows align POSITIONALLY. Merging on createdTime instead
            # fans out on patients that carry duplicate timestamps (several do) and yields
            # more rows than the frame -> broadcast error. Merge only as a fallback, and
            # de-duplicate the key first so it cannot fan out.
            if len(calib) != len(df):
                calib = df[["createdTime"]].merge(
                    calib.drop_duplicates(subset="createdTime"), on="createdTime", how="left")
            if len(calib) != len(df):        # still mismatched -> refuse rather than corrupt
                calib = None
        if calib is not None:
            armed = (calib["armed"].fillna(0).to_numpy() == 1)
            seg = calib["calib_segment"].ffill().fillna(0).to_numpy().astype(int)
            candidate = calib["candidate_event"].fillna(0).to_numpy().astype(int)
            # per-segment priors (m0, sigma0) — constant within a segment, hence within a chunk
            seg_priors = {int(s): {"m0": float(g["calib_m0"].iloc[0]),
                                   "sigma0": float(g["calib_sigma0"].iloc[0])}
                          for s, g in calib.groupby("calib_segment")}
            pop = float(pop_sigma0) if pop_sigma0 else POP_SIGMA0_DEFAULT
            default_priors = next(iter(seg_priors.values()), {"m0": 0.0, "sigma0": pop})

            def priors_for_chunk(idx):
                return seg_priors.get(int(seg[idx[0]]), default_priors)
            log_sigma0 = float(default_priors["sigma0"])
        else:
            # ---- Fallback: 03's original cumulative 24h burn-in (population prior) ----
            burn = resid[obs_idx[:BURN_IN_N]]             # first 144 VALID samples (cross-chunk)
            n = int(burn.size)
            pop = float(pop_sigma0) if pop_sigma0 else POP_SIGMA0_DEFAULT
            m_obs = float(np.median(burn))
            sig_obs = float(max(1.4826 * np.median(np.abs(burn - m_obs)), np.std(burn), 1e-3))
            sigma0 = float(np.sqrt((POP_N0 * pop ** 2 + n * sig_obs ** 2) / (POP_N0 + n)))
            m0 = float((POP_N0 * 0.0 + n * m_obs) / (POP_N0 + n))
            _priors = {"m0": m0, "sigma0": sigma0}
            candidate = np.zeros(len(df), int)
            armed = np.zeros(len(df), bool)
            if obs_idx.size > BURN_IN_N:
                armed[obs_idx[BURN_IN_N:]] = True

            def priors_for_chunk(idx):
                return _priors
            log_sigma0 = sigma0

        # ---- per-method, per-chunk detection on observed residuals --------------
        per = {name: {"anomaly": np.zeros(len(df), int),
                      "phenotype": np.full(len(df), "normal", dtype=object),
                      "magnitude": np.zeros(len(df)),
                      "severity": np.zeros(len(df))} for name, _ in METHODS}

        for cid in df["chunk_id"].unique():
            sel = observed & (df["chunk_id"].to_numpy() == cid)
            idx = np.where(sel)[0]
            if idx.size < 2:
                continue
            t_chunk = elapsed[idx] - elapsed[idx[0]]
            v_chunk = resid[idx]
            chunk_priors = priors_for_chunk(idx)
            for name, module in METHODS:
                r = module.detect(t_chunk, v_chunk, chunk_priors)
                for k in ("anomaly", "phenotype", "magnitude", "severity"):
                    per[name][k][idx] = r[k]

        # ---- write per-method rich columns + ensemble (any-of-4) ---------------
        for name, _ in METHODS:
            a = per[name]["anomaly"] & armed                  # suppress during burn-in
            df[f"{name}_anomaly"] = a.astype(int)
            ph = per[name]["phenotype"].copy()
            ph[~armed] = "burn_in"
            ph[(a == 0) & armed] = "normal"
            df[f"{name}_phenotype"] = ph
            df[f"{name}_magnitude"] = per[name]["magnitude"]

        anomly = np.zeros(len(df), int)
        primary = np.full(len(df), "normal", dtype=object)
        primary[~armed] = "burn_in"
        mag = np.zeros(len(df))
        for r in range(len(df)):
            if not armed[r]:
                continue
            best_sev, best = -1.0, None
            for name, _ in METHODS:
                if per[name]["anomaly"][r] == 1:
                    anomly[r] = 1
                    sev = per[name]["severity"][r]
                    if sev > best_sev:
                        best_sev, best = sev, name
            if best is not None:
                primary[r] = per[best]["phenotype"][r]
                mag[r] = per[best]["magnitude"][r]
        df["anomaly_present"] = anomly
        df["primary_phenotype"] = primary
        df["magnitude"] = mag
        df["candidate_event"] = candidate      # post-gap keep-but-verify flag (0 in fallback mode)
        df["cpd_input_col"] = "residual_hrv"

        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = file_path.stem + "_annotated.csv"
        df.to_csv(out_dir / out_name, index=False)

        # Per-method EVENT counts (contiguous runs) for apples-to-apples reporting.
        def _events(a):
            a = np.asarray(a).astype(int)
            return int(((a == 1) & (np.r_[0, a[:-1]] == 0)).sum())
        ev = {f"{n}_events": _events(df[f"{n}_anomaly"]) for n, _ in METHODS}

        return {"file": file_path.name, "status": "success", "n_rows": len(df),
                "n_armed": int(armed.sum()), "n_anomaly": int(anomly.sum()),
                "ensemble_events": _events(anomly), "sigma0": round(log_sigma0, 3), **ev}
    except Exception as e:
        return {"file": file_path.name, "status": "failed", "reason": str(e)}


def run_annotation(method="gammadglm"):
    if method not in SMOOTHERS:
        print(f"Unknown method '{method}'. Choose from: {list(SMOOTHERS)}")
        return
    in_dir = Path(SMOOTHERS[method][0])
    out_dir = Path(f"data/annotated_{method}")
    calib_dir = Path(f"data/calibrated_{method}")       # Stage 03a output (optional)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(in_dir.glob("*.csv"))
    if not csv_files:
        print(f"ERROR: No smoothed CSVs in {in_dir.resolve()} (run the smoother first)")
        return

    print(f"\n--- CPD ANNOTATION (residual) — Track A = {method} ---")
    print(f"Input:    {in_dir.resolve()}")
    print(f"Output:   {out_dir.resolve()}")
    pop_sigma0 = _population_sigma0(csv_files)            # population prior (cohort-derived)
    calib_on = calib_dir.exists() and any(calib_dir.glob("*_calib.csv"))
    print(f"Patients: {len(csv_files)}   Methods: {[m for m, _ in METHODS]}")
    print(f"CPD input column: residual_hrv   "
          f"Baseline: {'Stage-03a adaptive (' + str(calib_dir) + ')' if calib_on else f'inline {BURN_IN_N}-pt burn-in'}   "
          f"Pop prior σ0={pop_sigma0:.2f} (N0={POP_N0})   Vote: any-of-4\n")

    results = [annotate_patient(f, out_dir, pop_sigma0, calib_dir if calib_on else None)
               for f in tqdm(csv_files, desc="Annotating")]
    df_log = pd.DataFrame(results)
    df_log.to_csv(RESULTS_DIR / f"annotation_{method}_log.csv", index=False)

    ok = df_log[df_log["status"] == "success"]; bad = df_log[df_log["status"] == "failed"]
    print("\n" + "=" * 60 + f"\nANNOTATION COMPLETE — {method}\n" + "=" * 60)
    print(f"Success: {len(ok)} / {len(csv_files)}   Failed: {len(bad)}")
    print(f"Output:  {out_dir.resolve()}")
    if len(bad) > 0:
        for _, row in bad.iterrows():
            print(f"  - {row['file']}: {row['reason']}")
    samp = sorted(out_dir.glob("*_annotated.csv"))
    if samp:
        print(f"\nOutput columns: {pd.read_csv(samp[0], nrows=0).columns.tolist()}")


if __name__ == "__main__":
    run_annotation(sys.argv[1] if len(sys.argv) > 1 else "gammadglm")
