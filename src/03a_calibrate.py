"""Stage 03a — Adaptive baseline calibration + gap-aware carry/restart.

Sits BETWEEN the smoother (02x) and the CPD ensemble (03). It replaces 03's
one-shot 144-sample burn-in with a data-driven baseline that GROWS while the
circadian-detrended residual stays stationary, FREEZES when the estimate stops
moving, and CAPS at the drift horizon — and it decides, per gap, whether the
baseline carries across the silence or restarts.

Nothing downstream is required to change: 03 uses this stage's output IF the
`data/calibrated_<method>/` file exists, and otherwise falls back to its own
inline burn-in. Running 03a is therefore purely additive.

Strict causality (H9): every quantity at row i is computed from observed rows
<= i only (forward pass, no peeking). `python src/03a_calibrate.py selftest`
asserts the truncation guarantee (prefix run == full run on shared rows) plus
the gap carry/restart boundaries.

Per-patient output columns (same grid/order as the smoothed input):
  calib_segment    int    increments on every burn-in RESTART
  calib_m0         float  segment baseline mean fed to the detectors as prior
  calib_sigma0     float  segment noise floor (robust std, pop-shrunk) — the prior
  calib_sigma_infl float  sigma0 inflated for uncertainty during silence (inspection)
  armed            0/1    1 once the segment has >= BURN_IN_FLOOR valid samples
  candidate_event  0/1    first returning point after a keep-but-verify gap
                          (a post-gap shift is FLAGGED, not silently absorbed)

Usage (from asthma-prediction/):
    python src/03a_calibrate.py [method]     # method default: gammadglm
    python src/03a_calibrate.py selftest     # assert H9 + gap logic on synthetic data
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Track-A smoother -> (smoothed dir, file suffix). Mirrors SMOOTHERS in 03_annotate.py.
SMOOTHERS = {
    "pf":        ("data/smoothed",           "_smoothed"),
    "rspf":      ("data/smoothed_rspf",      "_rspf"),
    "krlst":     ("data/smoothed_krlst",     "_krlst"),
    "gpssm":     ("data/smoothed_gpssm",     "_gpssm"),
    "ossa":      ("data/smoothed_ossa",      "_ossa"),
    "kim":       ("data/smoothed_kim",       "_kim"),
    "gammadglm": ("data/smoothed_gammadglm", "_gammadglm"),
    "logmhw":    ("data/smoothed_logmhw",    "_logmhw"),
    "hrkf":      ("data/smoothed_hrkf",      "_hrkf"),
    "anrewma":   ("data/smoothed_anrewma",   "_anrewma"),
    "ctimm":     ("data/smoothed_ctimm",     "_ctimm"),
}

PER_DAY = 144                       # 10-min grid: one circadian cycle
BURN_IN_FLOOR = 144                 # >=24h valid samples before a segment ARMS (matches 03's BURN_IN_N)
DRIFT_HORIZON = 21 * PER_DAY        # baseline stops growing at ~21 days even if never stationary
MEMORY_CAP = 90 * PER_DAY           # effective memory ceiling (oldest points drop off)

GAP_KEEP_MIN = 24 * 60.0           # < 1 day: carry the baseline silently
GAP_RESTART_MIN = 21 * 24 * 60.0   # >= 21 days (or no valid reference): restart burn-in
#   GAP_KEEP_MIN <= gap < GAP_RESTART_MIN : keep-but-verify (flag the returning point)

STAT_EPS = 0.02                     # rel. change of sigma0 below which the estimate is "not moving"
STAT_WINDOW = PER_DAY // 6          # sustained for this many growth steps (~4h) -> freeze
TAU_INFLATE_MIN = GAP_KEEP_MIN      # silence time-constant for uncertainty inflation

POP_N0 = 30                         # population-prior pseudo-count (Bayesian shrinkage strength)
POP_SIGMA0_DEFAULT = 15.0           # fallback population noise prior (ms) if cohort unavailable

GAP_THRESHOLD_MIN = 180.0          # H8 chunk flush — kept for parity; restarts are a superset
RESULTS_DIR = Path("./data/results")


def _population_sigma0(csv_files):
    """Cohort noise prior = median per-patient robust std of residual_hrv.
    ponytail: duplicated from 03_annotate.py — a module named '03_annotate' isn't importable."""
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


def _robust_floor(detr, pop_sigma0):
    """(m0, sigma0) from detrended residuals, variance shrunk toward the population
    prior by pseudo-count POP_N0; mean shrunk toward 0 (residual baseline)."""
    n = int(detr.size)
    m_obs = float(np.median(detr))
    sig_obs = float(max(1.4826 * np.median(np.abs(detr - m_obs)), np.std(detr), 1e-3))
    sigma0 = float(np.sqrt((POP_N0 * pop_sigma0 ** 2 + n * sig_obs ** 2) / (POP_N0 + n)))
    m0 = float((n * m_obs) / (POP_N0 + n))
    return m0, sigma0


def calibrate_arrays(times_min, resid, tod_bin, pop_sigma0):
    """Core causal forward pass over one patient. Pure arrays so the self-test can
    drive it directly. `times_min` = minutes since patient start (per row); `resid`
    NaN where unobserved; `tod_bin` = time-of-day bin 0..143.

    Returns per-row (calib_segment, calib_m0, calib_sigma0, calib_sigma_infl,
    armed, candidate_event). Unobserved rows inherit the current segment state.
    """
    N = len(resid)
    seg = np.zeros(N, int)
    m0_col = np.full(N, np.nan)
    s0_col = np.full(N, np.nan)
    sinfl_col = np.full(N, np.nan)
    armed = np.zeros(N, np.int8)
    candidate = np.zeros(N, np.int8)

    # circadian detrend, causal: running per-bin mean of residual seen SO FAR (H9).
    bin_sum = np.zeros(PER_DAY)
    bin_cnt = np.zeros(PER_DAY, int)

    seg_id = 0
    hist = []                       # detrended residuals in the current segment (<= MEMORY_CAP)
    m0 = 0.0
    sigma0 = pop_sigma0
    frozen = False
    stat_run = 0                    # consecutive near-stationary growth steps
    prev_sigma = None
    n_valid = 0                     # observed samples in current segment
    last_obs_t = None               # minutes of previous observed row

    def _reset_segment():
        nonlocal hist, m0, sigma0, frozen, stat_run, prev_sigma, n_valid
        hist = []
        m0, sigma0 = 0.0, pop_sigma0
        frozen = False
        stat_run = 0
        prev_sigma = None
        n_valid = 0

    _reset_segment()

    for i in range(N):
        v = resid[i]
        is_candidate = False

        if np.isfinite(v):
            gap = (times_min[i] - last_obs_t) if last_obs_t is not None else 0.0
            # --- gap decision (carry vs verify vs restart) --------------------
            if last_obs_t is not None and gap >= GAP_RESTART_MIN:
                seg_id += 1
                _reset_segment()
            elif last_obs_t is not None and gap >= GAP_KEEP_MIN:
                is_candidate = True          # keep-but-verify: flag the returning point
            last_obs_t = times_min[i]

            # --- circadian detrend using only past+current bin stats (causal) --
            b = int(tod_bin[i])
            detr_i = v - (bin_sum[b] / bin_cnt[b]) if bin_cnt[b] > 0 else v
            bin_sum[b] += v
            bin_cnt[b] += 1

            n_valid += 1
            # --- grow / freeze the baseline -----------------------------------
            if not frozen:
                hist.append(detr_i)
                if len(hist) > MEMORY_CAP:
                    hist.pop(0)
                m0, sigma0 = _robust_floor(np.asarray(hist), pop_sigma0)
                if prev_sigma is not None and prev_sigma > 0:
                    if abs(sigma0 - prev_sigma) / prev_sigma < STAT_EPS:
                        stat_run += 1
                    else:
                        stat_run = 0
                prev_sigma = sigma0
                if stat_run >= STAT_WINDOW or n_valid >= DRIFT_HORIZON:
                    frozen = True        # estimate stopped moving, or hit drift horizon
            else:
                # frozen: baseline of "normal" survives; keep circadian detrend live
                pass

            # --- uncertainty inflation during silence (inspection column) ------
            infl = sigma0 * np.sqrt(1.0 + gap / TAU_INFLATE_MIN) if gap > 0 else sigma0

            armed[i] = 1 if n_valid >= BURN_IN_FLOOR else 0
            candidate[i] = 1 if is_candidate else 0
            sinfl_col[i] = infl
        else:
            # unobserved row: inherit segment state, never armed, no detection
            sinfl_col[i] = sigma0

        seg[i] = seg_id
        m0_col[i] = m0
        s0_col[i] = sigma0

    return seg, m0_col, s0_col, sinfl_col, armed, candidate


def calibrate_patient(file_path, out_dir, pop_sigma0):
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        if "residual_hrv" not in df.columns:
            return {"file": file_path.name, "status": "failed", "reason": "no residual_hrv column"}
        df["createdTime"] = pd.to_datetime(df["createdTime"])
        df = df.sort_values("createdTime").reset_index(drop=True)

        resid = pd.to_numeric(df["residual_hrv"], errors="coerce").to_numpy()
        t = (df["createdTime"] - df["createdTime"].iloc[0]).dt.total_seconds().to_numpy() / 60.0
        tod = (df["createdTime"].dt.hour * 6 + df["createdTime"].dt.minute // 10).to_numpy()

        seg, m0, s0, sinfl, armed, cand = calibrate_arrays(t, resid, tod, pop_sigma0)

        out = df[["createdTime"]].copy()
        out["calib_segment"] = seg
        out["calib_m0"] = m0
        out["calib_sigma0"] = s0
        out["calib_sigma_infl"] = sinfl
        out["armed"] = armed
        out["candidate_event"] = cand

        out_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_dir / (file_path.stem + "_calib.csv"), index=False)
        return {"file": file_path.name, "status": "success", "n_rows": len(df),
                "n_segments": int(seg.max() + 1), "n_armed": int(armed.sum()),
                "n_candidate": int(cand.sum())}
    except Exception as e:
        return {"file": file_path.name, "status": "failed", "reason": str(e)}


def run_calibration(method="gammadglm"):
    if method not in SMOOTHERS:
        print(f"Unknown method '{method}'. Choose from: {list(SMOOTHERS)}")
        return
    in_dir = Path(SMOOTHERS[method][0])
    out_dir = Path(f"data/calibrated_{method}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(in_dir.glob("*.csv"))
    if not csv_files:
        print(f"ERROR: No smoothed CSVs in {in_dir.resolve()} (run the smoother first)")
        return

    pop_sigma0 = _population_sigma0(csv_files)
    print(f"\n--- ADAPTIVE BASELINE CALIBRATION — Track A = {method} ---")
    print(f"Input:    {in_dir.resolve()}")
    print(f"Output:   {out_dir.resolve()}")
    print(f"Patients: {len(csv_files)}   Pop prior σ0={pop_sigma0:.2f} (N0={POP_N0})   "
          f"floor={BURN_IN_FLOOR}  horizon={DRIFT_HORIZON}  restart≥{GAP_RESTART_MIN/1440:.0f}d\n")

    results = [calibrate_patient(f, out_dir, pop_sigma0)
               for f in tqdm(csv_files, desc="Calibrating")]
    df_log = pd.DataFrame(results)
    df_log.to_csv(RESULTS_DIR / f"calibration_{method}_log.csv", index=False)
    ok = df_log[df_log["status"] == "success"]
    print("\n" + "=" * 60 + f"\nCALIBRATION COMPLETE — {method}\n" + "=" * 60)
    print(f"Success: {len(ok)} / {len(csv_files)}   Output: {out_dir.resolve()}")
    if len(ok):
        print(f"Segments/patient: median {int(ok['n_segments'].median())}  "
              f"candidate points: {int(ok['n_candidate'].sum())}")


# --------------------------------------------------------------------------- #
def _selftest():
    """Assert H9 truncation guarantee + gap carry/restart boundaries on synthetic data."""
    rng = np.random.default_rng(0)
    n = 600
    t = np.arange(n) * 10.0                      # 10-min grid, no gaps
    resid = rng.normal(0, 10, n)
    tod = (np.arange(n) * 10 // 10 % PER_DAY)
    pop = 10.0

    full = calibrate_arrays(t, resid, tod, pop)
    # H9: prefix run must equal full run on the shared prefix (no lookahead).
    k = 400
    pref = calibrate_arrays(t[:k], resid[:k], tod[:k], pop)
    for a, b in zip(pref, full):
        assert np.allclose(np.nan_to_num(a), np.nan_to_num(b[:k])), "H9 truncation violated"

    # Arming: disarmed before the floor, armed at/after it.
    armed = full[4]
    assert armed[:BURN_IN_FLOOR - 1].sum() == 0, "armed too early"
    assert armed[BURN_IN_FLOOR:].all(), "failed to arm after floor"

    # Gap logic: a <24h gap carries (same segment, no candidate); a >=21d gap restarts.
    t2 = t.copy(); t2[300:] += GAP_KEEP_MIN + 100          # ~1-day gap at row 300
    seg2, *_, cand2 = calibrate_arrays(t2, resid, tod, pop)
    assert seg2[300] == seg2[299], "keep-but-verify wrongly restarted"
    assert cand2[300] == 1 and cand2[301] == 0, "post-gap point not flagged as candidate"

    t3 = t.copy(); t3[300:] += GAP_RESTART_MIN + 100       # >21-day gap at row 300
    seg3 = calibrate_arrays(t3, resid, tod, pop)[0]
    assert seg3[300] == seg3[299] + 1, "long gap failed to restart segment"

    print("03a self-test OK — H9 truncation, arming floor, gap carry/verify/restart all hold.")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "gammadglm"
    if arg == "selftest":
        _selftest()
    else:
        run_calibration(arg)
