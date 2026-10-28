"""Smoothing study: derive every smoother's settings from the cohort, run, score, report.

    python scripts/smoothing_study.py stats       data characteristics per patient + cohort
    python scripts/smoothing_study.py derive      settings per method per cohort (+ reasons)
    python scripts/smoothing_study.py run         smooth + per-patient signal metrics
    python scripts/smoothing_study.py report      dashboard + methods page
    python scripts/smoothing_study.py all

Cohorts are AIIMS (adults + children) and COPD, the two with exacerbation records, and
both combined. Settings are derived on the cohort being smoothed. Output goes to
data/reports/smoothing/ (gitignored: it holds patient data).
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wearable.config import LABELS_CSV, PROCESSED_DIR, RAW_DIR, REPORTS_DIR  # noqa: E402

OUT = REPORTS_DIR / "smoothing"
PASS1 = RAW_DIR / "pass1"


def cohorts() -> dict[str, list[str]]:
    """Patients with processed HRV, by cohort. Group is kept for subgroup reporting."""
    idx = pd.read_csv(PASS1 / "patients.csv", dtype=str)
    idx = idx[idx["in_models"] == "1"]
    have = {p.stem.replace("_processed", "") for p in PROCESSED_DIR.glob("*_processed.csv")}
    idx = idx[idx["p_id"].isin(have)]
    out = {c: sorted(idx.loc[idx["cohort"] == c, "p_id"]) for c in ("aiims", "copd")}
    out["combined"] = sorted(out["aiims"] + out["copd"])
    return out


def patient_info() -> pd.DataFrame:
    idx = pd.read_csv(PASS1 / "patients.csv", dtype=str).set_index("p_id")
    lab = pd.read_csv(LABELS_CSV, dtype={"p_id": str}, parse_dates=["date"])
    exac = lab[lab["label"] == 1].groupby("p_id")["date"].apply(
        lambda s: sorted(d.strftime("%Y-%m-%d") for d in s))
    ids = cohorts()["combined"]
    info = idx.loc[ids, ["cohort", "group", "age", "sex", "file"]].copy()
    info["exac_days"] = [exac.get(p, []) for p in ids]
    info["has_exac"] = info["exac_days"].map(bool)
    return info


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------
def _one_stats(args):
    pid, rawfile = args
    from wearable import cohort_stats as cs
    df = pd.read_csv(PROCESSED_DIR / f"{pid}_processed.csv", parse_dates=["createdTime"])
    s = cs.patient_stats(df, pid)
    if s is None:
        return None
    raw = pd.read_csv(PASS1 / rawfile, usecols=["hrv", "sleep_active", "steps_active"])
    s.update(cs.state_medians(raw))
    return s


def cmd_stats() -> None:
    from wearable import cohort_stats as cs
    OUT.mkdir(parents=True, exist_ok=True)
    info = patient_info()
    with ProcessPoolExecutor() as pool:
        stats = [s for s in pool.map(_one_stats, list(info["file"].items())) if s is not None]
    by_id = {s["p_id"]: s for s in stats}
    (OUT / "patient_stats.json").write_text(json.dumps(stats, default=float))
    summary = {c: cs.summarise([by_id[p] for p in ids if p in by_id])
               for c, ids in cohorts().items()}
    (OUT / "cohort_stats.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"stats for {len(stats)} patients -> {OUT}")


# ---------------------------------------------------------------------------
# derive
# ---------------------------------------------------------------------------
# kind: derived (computed from this cohort) · physiology (fixed by biology / the device)
#       convention (standard statistical default, cited) · numerical (stability / cost,
#       does not change what is estimated) · structural (a modelling choice) · pipeline
#       (shared by every method, set in wearable/config.py)
KINDS = ("derived", "physiology", "convention", "numerical", "structural", "pipeline")


def _sig(x, n=4):
    if isinstance(x, (list, tuple)):
        return [_sig(v, n) for v in x]
    if isinstance(x, (bool, str, int, np.integer)) or x is None:
        return x
    return float(f"{x:.{n}g}")


def _gain(r):
    """Steady-state Kalman gain A of a local-level model with signal-to-noise q/R = r:
    A^2 / (1 - A) = r  (West & Harrison). A discount factor is then delta = 1 - A."""
    return 0.0 if r <= 0 else (-r + np.sqrt(r * r + 4 * r)) / 2


def derive_settings(c: dict, P: pd.DataFrame) -> dict[str, list[dict]]:
    """c = cohort summary, P = that cohort's per-patient stats. -> {method: [setting]}."""
    def pmed(series):
        s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        return float(s.median())

    dt = c["dt_min"]
    spd = 1440.0 / dt                                   # readings per day
    R, S, tau = c["R_log"], c["S_log"], c["tau_log"]
    phi = np.exp(-dt / tau)
    q_step = 2 * S * (1 - phi)                          # level change per reading (variance)
    r_snr = q_step / R
    A = _gain(r_snr)
    r_seas = (c.get("dcoef_var_log", 0.0) / spd) / R
    A_seas = _gain(r_seas)
    harm = [h for h, f in sorted(c["frac_harm_sig"].items()) if f >= 0.5]
    n_harm = int(max(harm)) if harm else 1
    tau_day = c["tau_day_min"]
    r_scale_log = pmed(P["R_log"] / P["module_noise_log"])
    vol = c["hour_vol_p10_p50_p90"]

    common = [
        dict(knob="gap_threshold_min", value=180.0, kind="pipeline",
             meaning="A gap at least this long splits the recording; the filter restarts.",
             effect="Higher: smooths across longer voids. Lower: more, shorter segments.",
             how=f"Shared H8 constant. In this cohort 99% of gaps inside a segment are "
                 f"<= {c['inside_dt_quantiles']['0.99']:g} min and the gap histogram has no "
                 f"natural break, so the data do not argue for moving it."),
        dict(knob="min_segment_rows", value=10, kind="pipeline",
             meaning="Segments with fewer readings are left unsmoothed.",
             effect="Higher: fewer, more reliable estimates. Lower: smooths tiny fragments.",
             how=f"Shared constant. {c['frac_readings_in_chunks_ge_10']:.1%} of readings sit "
                 f"in segments of >= 10."),
    ]
    ref_dt = dict(knob="ref_dt_min", value=dt, kind="derived",
                  meaning="Reference sampling step; elapsed time is measured in these units.",
                  effect="Must equal the real cadence, otherwise every rate is mis-scaled.",
                  how=f"Median gap between readings inside a segment = {dt:g} min.")
    harmonics = dict(knob="n_harmonics", value=n_harm, kind="derived",
                     meaning="Number of 24 h Fourier harmonics in the circadian term.",
                     effect="More: sharper daily shape, more parameters to learn per segment.",
                     how=f"Harmonic h kept if significant (p<0.05, autocorrelation-corrected F "
                         f"test) in >= 50% of patients: share significant "
                         f"{ {k: round(v, 2) for k, v in c['frac_harm_sig'].items()} }; 1-3 tested.")
    period = dict(knob="period_min", value=1440.0, kind="physiology",
                  meaning="Length of the circadian cycle.", effect="Fixed by biology.",
                  how="24 h.")
    deltrend = dict(knob="deltrend", value=1 - A, kind="derived",
                    meaning="Discount on the level: how much past information is kept per step.",
                    effect="Closer to 1: stiffer, smoother level. Lower: follows changes faster.",
                    how=f"1 - A, A = steady-state gain for signal-to-noise q/R = {r_snr:.3g} "
                        f"(level change per reading q = 2S(1-e^(-dt/tau)) = {q_step:.3g}, "
                        f"noise R = {R:.3g}, S = {S:.3g}, tau = {tau:.0f} min, log units).")
    delseas = dict(knob="delseas", value=1 - A_seas, kind="derived",
                   meaning="Discount on the circadian harmonics: how fast the daily shape may change.",
                   effect="1: shape fixed once learned. Lower: shape adapts within the day.",
                   how=f"1 - A for r = (day-to-day change of the harmonic coefficients / readings "
                       f"per day) / R = {r_seas:.3g}. Day-to-day change beyond estimation noise: "
                       f"{c.get('dcoef_var_log', 0.0):.3g}/day (0 = not detectable).")

    amps = [c[f"amp{h}_log"] for h in range(1, n_harm + 1)]
    priors = [
        dict(knob="C0_var", value=S, kind="derived",
             meaning="Prior variance of the level at the start of each segment.",
             effect="Higher: the first readings of a segment move the level more.",
             how=f"The level's own variance S = {S:.3g} (log): how far it can be from the patient's median."),
        dict(knob="C0_seas_var", value=[a * a / 2 for a in amps], kind="derived",
             meaning="Prior variance of each circadian harmonic's two coefficients, per segment.",
             effect="Higher: short segments can invent a large daily rhythm. The old value 1 let "
                    "level and harmonics swing against each other (28% of outputs hit the range limits).",
             how="amplitude^2 / 2 per harmonic (cos and sin share the amplitude), amplitudes "
                 + " / ".join(f"{a:.3g}" for a in amps) + " (cohort median, log)."),
    ]

    out = {}
    out["anrewma"] = common + [ref_dt,
        dict(knob="norm_alpha", value=1 - np.exp(-dt / tau_day), kind="derived",
             meaning="EWMA rate of the running mean/std used to normalise (the slow baseline).",
             effect="Higher: normaliser follows faster drift. Lower: steadier baseline.",
             how=f"1 - exp(-dt / tau_day), tau_day = {tau_day:.0f} min from the day-to-day "
                 f"correlation of daily medians (r = {c['daymedian_lag1_corr']:.3f}, "
                 f"{c['n_day_pairs']} day pairs)."),
        dict(knob="ewma_r", value=c["R_raw"] / (c["R_raw"] + c["S_raw"]), kind="derived",
             meaning="Measurement noise of the robust EWMA, in normalised (z) units.",
             effect="Higher (vs ewma_q): smoother output.",
             how=f"Noise share of the variance: R/(R+S) = {c['R_raw']:.3g}/({c['R_raw']:.3g}+"
                 f"{c['S_raw']:.3g}) (raw HRV units)."),
        dict(knob="ewma_q", value=2 * c["S_raw"] * (1 - np.exp(-dt / c["tau_raw"])) / (c["R_raw"] + c["S_raw"]),
             kind="derived", meaning="Process noise per reading of the robust EWMA (z units).",
             effect="Higher (vs ewma_r): follows the signal more closely.",
             how=f"Level change per reading 2S(1-e^(-dt/tau)) / (R+S), tau = {c['tau_raw']:.0f} min."),
        dict(knob="ewma_c", value=1.0, kind="convention",
             meaning="Soft outlier threshold of the IMQ weight, in z units.",
             effect="Lower: more readings treated as outliers and ignored.",
             how="WoLF default (Duran-Martin et al. 2024): c = 1."),
        dict(knob="winsor", value=3.0, kind="convention",
             meaning="|z| clip when updating the normaliser.", effect="Lower: more robust normaliser.",
             how="3 sigma rule."),
        dict(knob="norm_init_window", value=20, kind="numerical",
             meaning="Readings used to start the normaliser in each segment.",
             effect="Higher: steadier start. Note: the start uses up to 20 readings ahead.",
             how="Kept. Half of all segments are shorter than this, so it mostly equals the segment."),
        dict(knob="sigma_floor_frac", value=0.05, kind="numerical",
             meaning="Floor on the normaliser std.", effect="Guard only.", how="Kept."),
    ]
    out["ctimm"] = common + [
        dict(knob="r_scale", value=r_scale_log, kind="derived",
             meaning="Multiplies the method's own noise estimate (MAD of differences).",
             effect="Higher: smoother.",
             how="Median per patient of variogram noise R / the method's MAD noise, so the "
                 "noise it assumes equals the white noise actually measured."),
        dict(knob="q_scale", value=(q_step / dt) / R, kind="derived",
             meaning="Level process noise per minute, as a multiple of the noise R.",
             effect="Higher: follows the signal more closely. Lower: smoother.",
             how=f"(q per reading / dt) / R = ({q_step:.3g}/{dt:g})/{R:.3g}."),
        dict(knob="mode_mults", value=[vol[0] / vol[1], 1.0, vol[2] / vol[1]], kind="derived",
             meaning="Calm / normal / volatile modes: process-noise multipliers.",
             effect="Wider spread: adapts stiffness over a wider range.",
             how=f"10th, 50th and 90th percentile of hourly noise level across "
                 f"{c['n_patient_hours_vol']} patient-hours, divided by the median."),
        dict(knob="stickiness", value=np.exp(-dt / c["tau_vol"]), kind="derived",
             meaning="Probability of staying in the same mode at the next reading.",
             effect="Higher: modes persist longer.",
             how=f"exp(-dt / tau_vol), tau_vol = {c['tau_vol']:.0f} min: how long an hourly "
                 f"noise level stays correlated."),
        dict(knob="ref_dt_min", value=dt, kind="numerical",
             meaning="Declared but not used by the code (time enters in minutes).",
             effect="None.", how="-"),
    ]
    out["gammadglm"] = common + [ref_dt, harmonics, period, deltrend, delseas,
        dict(knob="shape_scale", value=pmed((1 / P["R_log"]) / P["gamma_shape_module"]), kind="derived",
             meaning="Multiplies the Gamma shape the method calibrates per patient "
                     "(shape = observation precision).",
             effect="Higher: readings trusted more -> follows them closely. Lower: smoother.",
             how="For Gamma noise Var(log y) ~ 1/shape, so the right shape is 1/R. Median per "
                 "patient of (1/R) / the method's own shape (median^2 / MAD-noise)."),
        *priors,
        dict(knob="q_floor / q_cap", value=[1e-6, 5.0], kind="numerical",
             meaning="Bounds on the linear-predictor variance.", effect="Guards only.", how="Kept."),
    ]
    out["gpssm"] = common + [
        dict(knob="r_scale", value=pmed(P["R_log"] / P["gpssm_R_module"]), kind="derived",
             meaning="Multiplies the method's own noise estimate.", effect="Higher: smoother.",
             how="Median per patient of variogram noise R / the method's MAD noise."),
        dict(knob="q_scale", value=pmed(P["S_log"] * (1 - P["gpssm_F"] ** 2) / P["gpssm_Q_module"]),
             kind="derived", meaning="Multiplies the method's own process-noise estimate.",
             effect="Higher: follows the signal more closely.",
             how="For an AR(1) level with variance S and learned retention F the innovation "
                 "variance is S(1-F^2). Median per patient of that / the method's own estimate "
                 f"(learned F median {pmed(P['gpssm_F']):.3f})."),
        dict(knob="f_max / gp_max_pairs / seed", value=[0.999, 800, 0], kind="numerical",
             meaning="Cap on learned F, GP subsample size, RNG seed.",
             effect="Stability / speed only.", how="Kept."),
    ]
    out["hrkf"] = common + [ref_dt, harmonics, period, deltrend, delseas,
        dict(knob="r_scale", value=r_scale_log, kind="derived",
             meaning="Multiplies the method's own noise estimate.", effect="Higher: smoother.",
             how="Median per patient of variogram noise R / the method's MAD noise."),
        dict(knob="huber_delta", value=1.345, kind="convention",
             meaning="Residual (in std) beyond which a reading is down-weighted.",
             effect="Lower: more readings treated as outliers.",
             how="Huber's 1.345: 95% efficiency when the noise is Gaussian."),
        *priors,
        dict(knob="n_irls", value=5, kind="numerical",
             meaning="IRLS iterations of the Huber solve.", effect="Convergence only.", how="Kept."),
    ]
    out["kim"] = common + [
        dict(knob="phi", value=phi, kind="derived",
             meaning="Retention of the level toward its regime mean per reading.",
             effect="Higher: level wanders longer before returning.",
             how=f"exp(-dt / tau) = exp(-{dt:g}/{tau:.0f})."),
        dict(knob="stickiness", value=phi, kind="derived",
             meaning="Probability of staying in the same HRV regime at the next reading.",
             effect="Higher: regimes persist longer.",
             how="Same correlation time as the level: exp(-dt / tau)."),
        dict(knob="q_scale", value=pmed(P["S_log"] * (1 - np.exp(-2 * dt / P["tau_log"])) / P["module_noise_log"]),
             kind="derived", meaning="Process noise as a multiple of the method's MAD noise.",
             effect="Higher: follows the signal more closely.",
             how="Median per patient of the AR(1) innovation variance S(1-phi^2) / MAD noise."),
        dict(knob="r_scale", value=r_scale_log, kind="derived",
             meaning="Observation noise as a multiple of the method's MAD noise.",
             effect="Higher: smoother.", how="Median per patient of variogram R / MAD noise."),
        dict(knob="var_ratio", value=c["noise_ratio_high_low"], kind="derived",
             meaning="Noise in the high-HRV regime relative to the low one.",
             effect="Only shifts how noisy each regime is assumed to be.",
             how="Median per patient of the robust noise of readings above vs below the "
                 "patient's median (low values reflect the device cap compressing high HRV)."),
    ]
    lscale = tau / np.sqrt(2)
    out["krlst"] = common + [
        dict(knob="lengthscale_min", value=lscale, kind="derived",
             meaning="RBF kernel lengthscale in minutes: how far apart readings still inform each other.",
             effect="Longer: smoother.",
             how=f"tau / sqrt(2): an RBF kernel's correlation reaches 1/e at sqrt(2) lengthscales; "
                 f"tau = {tau:.0f} min."),
        dict(knob="c", value=pmed(P["R_log"] / P["S_log"]), kind="derived",
             meaning="Noise-to-signal ratio.", effect="Higher: smoother.",
             how="Median per patient of R / S."),
        dict(knob="forgetting_lambda", value=np.exp(-dt / tau_day), kind="derived",
             meaning="Per-reading forgetting; < 1 lets the tracker follow a drifting baseline.",
             effect="Lower: forgets faster.", how=f"exp(-dt / tau_day), tau_day = {tau_day:.0f} min."),
        dict(knob="budget", value=int(np.ceil(3 * lscale / dt)), kind="derived",
             meaning="Dictionary size: past readings the tracker keeps.",
             effect="Larger: more memory, cost grows as size^2.",
             how="Readings within 3 lengthscales (older ones weigh < 1%): ceil(3 l / dt)."),
        dict(knob="forgetmode / jitter", value=["B2P", 1e-6], kind="numerical",
             meaning="Forgetting scheme (back-to-prior, the authors' recommendation), Gram regulariser.",
             effect="Stability only.", how="Kept."),
    ]
    out["logmhw"] = common + [ref_dt,
        dict(knob="optimize", value=False, kind="structural",
             meaning="If True, alpha/beta/phi are fitted per patient (statsmodels MLE).",
             effect="True = per-patient settings; reserved for the per-patient experiment.",
             how="Off: this run uses the cohort value of the same MLE."),
        dict(knob="alpha", value=c["holt_alpha"], kind="derived",
             meaning="Level smoothing weight of each new reading.", effect="Lower: smoother.",
             how=f"Median per patient of the damped-Holt MLE on the deseasonalised longest segment "
                 f"(the method's own optimiser). Cross-check, variogram gain A = {A:.3g}."),
        dict(knob="beta", value=c["holt_beta"], kind="derived",
             meaning="Trend smoothing weight.", effect="Higher: trend reacts faster.",
             how="Median per patient of the same MLE (0 = no trend needed)."),
        dict(knob="phi", value=c["holt_phi"], kind="derived",
             meaning="Trend damping per reading.", effect="Lower: trend dies out faster.",
             how="Median per patient of the same MLE."),
        dict(knob="gamma", value=_gain(c.get("dcoef_var_log", 0.0) / R), kind="derived",
             meaning="Weight of a new reading on its time-of-day seasonal value.",
             effect="0: daily profile fixed. Higher: profile adapts.",
             how="Each time-of-day bucket is updated once a day, so gain A for r = day-to-day "
                 f"change / R = {c.get('dcoef_var_log', 0.0) / R:.3g}."),
        dict(knob="buckets_per_day", value=int(round(spd)), kind="derived",
             meaning="Time-of-day buckets for the seasonal profile.", effect="One per reading slot.",
             how=f"1440 / dt = {spd:g}."),
    ]
    out["ossa"] = common + [
        dict(knob="window_max", value=int(round(spd)), kind="derived",
             meaning="Longest trailing window (readings) the SVD looks at.",
             effect="Longer: slower patterns resolved, more lag after a change.",
             how=f"One day of readings (1440 / dt), the longest physiological cycle. Note half of "
                 f"segments are <= {c['chunk_len_quantiles']['0.5']:g} readings, so the window is "
                 f"usually the whole segment so far."),
        dict(knob="n_components", value=int(round(c["ssa_rank_1day"])), kind="derived",
             meaning="Leading singular components kept as signal.", effect="Fewer: smoother.",
             how="Median per patient of the Gavish-Donoho optimal rank (known noise = variogram "
                 "R) of one-day Hankel windows."),
        dict(knob="l_frac", value=0.5, kind="convention",
             meaning="Embedding dimension as a fraction of the window.", effect="-",
             how="L = N/2 (Golyandina & Zhigljavsky)."),
        dict(knob="min_window", value=8, kind="numerical",
             meaning="Below this many readings, emit a trailing mean.", effect="Warm-up only.",
             how="Kept."),
    ]
    tl = c["tau_logit"]
    out["rspf"] = common + [
        dict(knob="K", value=c["n_distinct_states"], kind="derived",
             meaning="Number of HRV regimes.", effect="More: more baseline levels to switch between.",
             how="Activity states (sleep / rest / active, from the watch) whose HRV differs across "
                 "patients (paired Wilcoxon, Bonferroni): "
                 + ", ".join(f"{k} p={v['p']:.2g}" for k, v in c["state_tests"].items())),
        dict(knob="stickiness", value=np.exp(-dt / tl), kind="derived",
             meaning="Probability of staying in the same regime at the next reading.",
             effect="Higher: regimes persist longer.", how=f"exp(-dt / tau), logit tau = {tl:.0f} min."),
        dict(knob="reversion_tau_min", value=tl, kind="derived",
             meaning="Minutes for the level to revert to its regime mean.", effect="Longer: stiffer level.",
             how="Variogram correlation time in logit space."),
        dict(knob="sigma_level_frac", value=pmed(np.sqrt(P["S_logit"] * (1 - np.exp(-2 * dt / P["tau_logit"]))) / P["spread_logit"]),
             kind="derived", meaning="Level shock per 10 min, as a fraction of the patient's spread.",
             effect="Lower: smoother.",
             how="Median per patient of sqrt(S(1-phi^2)) / spread (logit units)."),
        dict(knob="sigma_seas", value=np.sqrt(c.get("dcoef_var_logit", 0.0) / spd), kind="derived",
             meaning="Per-step drift of the circadian coefficients.", effect="0: rhythm fixed.",
             how="sqrt(day-to-day change of the harmonic coefficients / readings per day)."),
        dict(knob="circ_init_frac", value=pmed(P["amp1_logit"] / P["spread_logit"]), kind="derived",
             meaning="Starting circadian amplitude relative to the spread.", effect="Prior only.",
             how="Median per patient of first-harmonic amplitude / spread."),
        dict(knob="n_particles / seed", value=[500, 0], kind="numerical",
             meaning="Monte Carlo particles, RNG seed.", effect="More: less Monte Carlo noise, slower.",
             how="Kept."),
        dict(knob="min_obs_sigma / min_logit_spread", value=[1.0, 0.2], kind="numerical",
             meaning="Floors.", effect="Guards only.", how="Kept."),
    ]
    for m, rows in out.items():
        for r in rows:
            r["value"] = _sig(r["value"])
    return out


# knob names the smoother configs accept (grouped rows like "a / b" are documentation)
def overrides(rows: list[dict]) -> dict:
    return {r["knob"]: (tuple(r["value"]) if isinstance(r["value"], list) else r["value"])
            for r in rows if r["kind"] in ("derived", "structural") and "/" not in r["knob"]}


def cmd_derive() -> None:
    summary = json.loads((OUT / "cohort_stats.json").read_text())
    P = pd.DataFrame(json.loads((OUT / "patient_stats.json").read_text())).set_index("p_id")
    settings = {}
    for coh, ids in cohorts().items():
        settings[coh] = derive_settings(summary[coh], P.loc[[p for p in ids if p in P.index]])
    (OUT / "settings.json").write_text(json.dumps(settings, indent=1))
    rows = [{"cohort": coh, "method": m, **r, "value": json.dumps(r["value"])}
            for coh, ms in settings.items() for m, rs in ms.items() for r in rs]
    pd.DataFrame(rows).to_csv(OUT / "settings.csv", index=False)
    from wearable import registry
    for m in settings["combined"]:
        defaults = registry.get("smoothing", m).defaults
        line = "  ".join(f"{k}={ {c: settings[c][m] and overrides(settings[c][m]).get(k) for c in settings} }"
                         for k in overrides(settings["combined"][m]))
        print(f"{m:10} default {defaults}\n           {line}")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
# A run is (variant, cohort): "derived" settings on the cohort they were derived from,
# plus the current registry defaults once on everyone (a patient's result does not
# depend on who else is in the run, so cohort numbers for defaults are a slice).
def runs() -> list[tuple[str, str]]:
    return [("default", "combined")] + [("derived", c) for c in ("aiims", "copd", "combined")]


def _patient_metrics(table: pd.DataFrame) -> pd.DataFrame:
    from wearable import metrics
    fns = {n: metrics.metric(n).load().compute for n in metrics.names("signal")}
    rows = []
    for pid, g in table.groupby("p_id", sort=True):
        r = {"p_id": str(pid), "n_smoothed": int(g["smoothed_hrv"].notna().sum())}
        r.update({n: float(f(g["hrvValue"], g["smoothed_hrv"])) for n, f in fns.items()})
        rows.append(r)
    return pd.DataFrame(rows)


def cmd_run() -> None:
    from wearable import smoothing
    from wearable.steps import evaluate, smooth
    settings = json.loads((OUT / "settings.json").read_text())
    ids = cohorts()
    per, log = [], []
    for method in smoothing.names():
        for variant, coh in runs():
            kw = {} if variant == "default" else overrides(settings[coh][method])
            art = smooth.run(method, patients=ids[coh], **kw)
            evaluate.run(art)                          # cohort-level metrics artifact too
            m = _patient_metrics(art.read_table("smoothed"))
            m.insert(0, "run_cohort", coh)
            m.insert(0, "variant", variant)
            m.insert(0, "method", method)
            per.append(m)
            st = art.manifest.get("stats", {})
            log.append({"method": method, "variant": variant, "cohort": coh,
                        "key": art.key[:12], "dir": str(art.dir),
                        "n_patients": st.get("n_patients"), "n_failed": st.get("n_failed"),
                        "params": json.dumps(art.manifest["params"]),
                        "seconds": (pd.Timestamp(art.manifest["finished"]) -
                                    pd.Timestamp(art.manifest["created"])).total_seconds()})
    per = pd.concat(per, ignore_index=True)
    info = patient_info()
    per = per.merge(info[["cohort", "group", "has_exac"]], left_on="p_id", right_index=True)
    per.to_csv(OUT / "metrics_per_patient.csv", index=False)
    pd.DataFrame(log).to_csv(OUT / "runs.csv", index=False)
    summarise_metrics(per).to_csv(OUT / "metrics_summary.csv", index=False)
    print(f"-> {OUT}/metrics_per_patient.csv, metrics_summary.csv, runs.csv")


def summarise_metrics(per: pd.DataFrame, names: list[str] | None = None) -> pd.DataFrame:
    """Median (and IQR) across patients, for each method x setting x group of patients.
    'setting' says where the settings came from: default, derived on own cohort, or
    derived on the combined cohort."""
    from wearable import metrics
    names = names or metrics.names("signal")
    groups = {"aiims": per["cohort"] == "aiims", "copd": per["cohort"] == "copd",
              "aiims_adult": per["group"] == "aiims_adult", "aiims_paed": per["group"] == "aiims_paed",
              "combined": per["cohort"].notna(), "exacerbation_patients": per["has_exac"]}
    rows = []
    for (method, variant, rc), g in per.groupby(["method", "variant", "run_cohort"]):
        setting = ("default" if variant == "default" else
                   "derived_combined" if rc == "combined" else "derived_own")
        for gname, mask in groups.items():
            sub = g[mask.loc[g.index]]
            if gname in ("aiims", "aiims_adult", "aiims_paed") and rc == "copd":
                continue
            if gname == "copd" and rc == "aiims":
                continue
            if rc != "combined" and gname in ("combined", "exacerbation_patients"):
                continue
            if not len(sub):
                continue
            r = {"method": method, "setting": setting, "patients": gname, "n": len(sub)}
            for n in names:
                v = sub[n].dropna()
                r[n] = float(v.median()) if len(v) else np.nan
                r[f"{n}_q25"], r[f"{n}_q75"] = (float(v.quantile(.25)), float(v.quantile(.75))) if len(v) else (np.nan, np.nan)
            rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DASH = OUT / "dashboard"
SETTING_LABEL = {"default": "Default settings", "own": "Derived from own cohort",
                 "combined": "Derived from AIIMS + COPD"}
TRUNCATION_DEV = {"anrewma": 0.0, "ctimm": 0.0, "hrkf": 0.002, "gammadglm": 0.052,
                  "ossa": 0.111, "kim": 0.122, "krlst": 0.135, "gpssm": 0.172,
                  "rspf": 0.388, "logmhw": 0.479}       # from `wearable selftest`


def _setting_of(variant: str, run_cohort: str, patient_cohort: str) -> str | None:
    if variant == "default":
        return "default"
    if run_cohort == "combined":
        return "combined"
    return "own" if run_cohort == patient_cohort else None


def _js(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), allow_nan=False, default=float)


def _clean(x):
    """NaN/inf -> None, recursively, so the JSON is valid."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


ABSOLUTE = ["var_raw", "var_sm", "rough_raw", "rough_sm", "mean_raw", "mean_sm",
            "peak_raw", "peak_sm", "std_raw", "resid_std"]


def _absolute(raw, smoothed) -> dict:
    """The raw and smoothed quantities behind the % metrics, in raw HRV units (variance squared), on
    the same rows the metrics use. peak_*: median excursion over the raw peaks that
    peak_preservation scores, in each signal."""
    from scipy.signal import find_peaks
    from wearable.metrics.peak_preservation import WINDOW
    raw, sm = np.asarray(raw, dtype=float), np.asarray(smoothed, dtype=float)
    m = np.isfinite(raw) & np.isfinite(sm)
    if m.sum() < 3:
        return {}
    raw, sm = raw[m], sm[m]
    peaks, _ = find_peaks(raw, prominence=0.5 * np.std(raw))
    win = [slice(max(0, p - WINDOW), p + WINDOW + 1) for p in peaks]
    return {"var_raw": np.var(raw), "var_sm": np.var(sm),
            "rough_raw": np.mean(np.abs(np.diff(raw))), "rough_sm": np.mean(np.abs(np.diff(sm))),
            "mean_raw": np.mean(raw), "mean_sm": np.mean(sm),
            "peak_raw": np.median([raw[p] - raw[w].min() for p, w in zip(peaks, win)]) if len(peaks) else np.nan,
            "peak_sm": np.median([sm[p] - sm[w].min() for p, w in zip(peaks, win)]) if len(peaks) else np.nan,
            "std_raw": np.std(raw), "resid_std": np.std(raw - sm)}


def ideal_reference(c: dict, n: int = 50000, seed: int = 0) -> dict:
    """What the best possible causal smoother would score, if HRV were exactly this
    cohort's model: an AR(1) level (variance S, correlation time tau) plus white noise R,
    in raw HRV units, one reading every dt. The exact Kalman filter is that smoother.
    No gaps, cap or circadian rhythm, so it is a yardstick, not a target."""
    from wearable import metrics
    rng = np.random.default_rng(seed)
    R, S, tau, dt = c["R_raw"], c["S_raw"], c["tau_raw"], c["dt_min"]
    phi, q = np.exp(-dt / tau), S * (1 - np.exp(-2 * dt / tau))
    x = np.empty(n); x[0] = rng.normal(0, np.sqrt(S))
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, np.sqrt(q))
    y = c["median_hrv"] + x + rng.normal(0, np.sqrt(R), n)
    m, P, est = y[0] - c["median_hrv"], R, np.empty(n)
    est[0] = m
    for i in range(1, n):
        mp, Pp = phi * m, phi * phi * P + q
        K = Pp / (Pp + R)
        m, P = mp + K * (y[i] - c["median_hrv"] - mp), (1 - K) * Pp
        est[i] = m
    est += c["median_hrv"]
    out = {name: float(metrics.metric(name).load().compute(y, est)) for name in metrics.names("signal")}
    out["true_level_corr"] = float(np.sqrt(S / (R + S)))
    return out


def cmd_report() -> None:
    import shutil
    import plotly
    from wearable import metrics, registry, store
    DASH.mkdir(parents=True, exist_ok=True)
    (DASH / "patients").mkdir(exist_ok=True)
    shutil.copy(Path(plotly.__file__).parent / "package_data" / "plotly.min.js", DASH / "plotly.min.js")

    info = patient_info()
    runs_df = pd.read_csv(OUT / "runs.csv")
    per = pd.read_csv(OUT / "metrics_per_patient.csv", dtype={"p_id": str})
    summary = pd.read_csv(OUT / "metrics_summary.csv")
    cstats = json.loads((OUT / "cohort_stats.json").read_text())
    settings = json.loads((OUT / "settings.json").read_text())
    mnames = metrics.names("signal")
    methods = list(dict.fromkeys(runs_df["method"]))

    # ---- per-patient series files -------------------------------------------
    tables = {}
    for _, r in runs_df.iterrows():
        art = store.load("smooth", r["method"], _full_key(r["method"], r["key"]))
        t = art.read_table("smoothed")
        t = t[t["hrvValue"].notna()].copy()
        t["p_id"] = t["p_id"].astype(str)
        tables[(r["method"], r["variant"], r["cohort"])] = {p: g for p, g in t.groupby("p_id")}
    for pid, row in info.iterrows():
        base = None
        series: dict = {}
        for (method, variant, rc), by_p in tables.items():
            s = _setting_of(variant, rc, row["cohort"])
            g = by_p.get(pid)
            if s is None or g is None:
                continue
            g = g.sort_values("createdTime")
            if base is None:
                base = g
            v = pd.Series(g["smoothed_hrv"].to_numpy(), index=g["createdTime"]).reindex(base["createdTime"])
            series.setdefault(method, {})[s] = [None if not np.isfinite(x) else round(float(x), 1) for x in v]
        if base is None:
            continue
        ts = pd.to_datetime(base["createdTime"])
        data = {"id": pid, "t0": ts.iloc[0].strftime("%Y-%m-%dT%H:%M:%S"),
                "t": ((ts - ts.iloc[0]).dt.total_seconds() / 60).round().astype(int).tolist(),
                "raw": base["hrvValue"].round(1).tolist(), "chunk": base["chunk_id"].astype(int).tolist(),
                "exac": row["exac_days"], "s": series}
        (DASH / "patients" / f"{pid}.js").write_text(f"window.__patient({_js(_clean(data))});")

    # ---- summary file ----------------------------------------------------------
    pp: dict = {}
    for _, r in per.iterrows():
        s = _setting_of(r["variant"], r["run_cohort"], r["cohort"])
        if s:
            pp.setdefault(r["p_id"], {}).setdefault(r["method"], {})[s] = {m: r[m] for m in mnames}
    absolute = pd.DataFrame([{"method": method, "variant": variant, "run_cohort": rc, "p_id": pid,
                              **_absolute(g["hrvValue"], g["smoothed_hrv"])}
                             for (method, variant, rc), by_p in tables.items() for pid, g in by_p.items()] +
                            # raw alone, on every reading, once per run so it slices like the smoothers
                            [{"method": "raw", "variant": variant, "run_cohort": rc, "p_id": pid,
                              **_absolute(g["hrvValue"], g["hrvValue"])}
                             for (method, variant, rc), by_p in tables.items() if method == methods[0]
                             for pid, g in by_p.items()])
    absolute = absolute.merge(info[["cohort", "group", "has_exac"]], left_on="p_id", right_index=True)
    first = tables[next(iter(tables))]
    patients = [{"id": pid, "cohort": row["cohort"], "group": row["group"],
                 "n": int(len(first[pid])) if pid in first else 0,
                 "exac": len(row["exac_days"])} for pid, row in info.iterrows()
                if (DASH / "patients" / f"{pid}.js").exists()]
    ref = {c: {"noise_share_raw_pct": 100 * cstats[c]["R_raw"] / (cstats[c]["R_raw"] + cstats[c]["S_raw"]),
               "ideal": ideal_reference(cstats[c])} for c in cstats}
    (OUT / "ideal_reference.json").write_text(json.dumps(ref, indent=1))
    meta = [{"name": n, "label": getattr(metrics.metric(n).load(), "LABEL", n),
             "hib": metrics.higher_is_better(n)} for n in mnames]
    labels = {m: registry.get("smoothing", m).label for m in methods}
    out = {"patients": patients, "perPatient": pp, "summary": summary.to_dict("records"),
           "absolute": summarise_metrics(absolute, ABSOLUTE).to_dict("records"),
           "metrics": meta, "methods": [{"key": m, "label": labels[m], "trunc": TRUNCATION_DEV.get(m)} for m in methods],
           "settingLabels": SETTING_LABEL, "reference": ref, "cohortStats": cstats, "overview": _overview(info),
           "runs": runs_df.to_dict("records")}
    (DASH / "summary.js").write_text(f"window.__summary = {_js(_clean(out))};")
    shutil.copy(HERE / "smoothing_dashboard.html", DASH / "index.html")

    page = (HERE / "smoothing_methods.html").read_text()
    payload = {"settings": settings, "cohortStats": {c: {k: v for k, v in s.items() if not k.endswith("__n")}
                                                     for c, s in cstats.items()},
               "labels": labels, "trunc": TRUNCATION_DEV,
               "defaults": {m: _config_defaults(m) for m in methods},
               "distance": _distances(summary, ref)}
    (OUT / "methods.html").write_text(page.replace("/*__DATA__*/null", _js(_clean(payload))))
    print(f"dashboard -> {DASH / 'index.html'}\nmethods   -> {OUT / 'methods.html'}")


def _overview(info: pd.DataFrame) -> dict:
    """Participant counts and the segmentation numbers behind the Data tab's flow chart,
    per group and in total. Segments are cut on observed readings, as the smoothers do."""
    from wearable.chunking import assign_chunks
    from wearable.config import MIN_SEGMENT_ROWS
    idx = pd.read_csv(PASS1 / "patients.csv", dtype=str)
    idx = idx[idx["in_models"] == "1"].set_index("p_id")
    seg = {}
    for pid in info.index:
        df = pd.read_csv(PROCESSED_DIR / f"{pid}_processed.csv", parse_dates=["createdTime"])
        df = df[df["hrvValue"].notna()].assign(patient_id=pid)
        n = assign_chunks(df, "createdTime", "patient_id").groupby("chunk_id").size()
        seg[pid] = (len(df), len(n), int(n[n >= MIN_SEGMENT_ROWS].sum()))
    out = {}
    for g in ("aiims_adult", "aiims_paed", "copd", "all"):
        ids = idx.index if g == "all" else idx.index[idx["group"] == g]
        hrv = [p for p in ids if p in seg]
        sm = [p for p in hrv if seg[p][2] > 0]
        ex = [p for p in sm if info.at[p, "has_exac"]]
        out[g] = {"participants": len(ids), "with_hrv": len(hrv), "smoothed": len(sm), "with_exac": len(ex),
                  "exac_days": int(sum(len(info.at[p, "exac_days"]) for p in ex)),
                  "readings": sum(seg[p][0] for p in hrv), "segments": sum(seg[p][1] for p in hrv),
                  "kept": sum(seg[p][2] for p in hrv),
                  "days_median": float(pd.to_numeric(idx.loc[hrv, "hrv_days"]).median())}
    return out


def _distances(summary: pd.DataFrame, ref: dict) -> dict:
    """{cohort: {method: {default, derived}}}: mean |gap| to the ideal causal filter over the
    four shape metrics on a 0-1 scale (same formula as the dashboard's cohort tab)."""
    scale = {"variance_reduction": 100, "roughness_reduction": 100, "peak_preservation": 1, "correlation": 1}
    out: dict = {}
    for c in ("aiims", "copd", "combined"):
        ideal = ref[c]["ideal"]
        for setting, col in (("default", "default"), ("derived_combined" if c == "combined" else "derived_own", "derived")):
            for _, r in summary[(summary["patients"] == c) & (summary["setting"] == setting)].iterrows():
                d = sum(abs(r[k] - ideal[k]) / s for k, s in scale.items()) / len(scale)
                out.setdefault(c, {}).setdefault(r["method"], {})[col] = round(float(d), 3)
    return out


def _config_defaults(method: str) -> dict:
    """Every knob's default: the config dataclass with the registry DEFAULTS applied."""
    import dataclasses
    from wearable import registry
    cfg = registry.get("smoothing", method).config()
    return dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else dict(cfg)


def _full_key(method: str, short: str) -> str:
    from wearable import store
    return json.loads((store.artifact_dir("smooth", method, short) / "manifest.json").read_text())["key"]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    steps = {"stats": [cmd_stats], "derive": [cmd_derive], "run": [cmd_run],
             "report": [cmd_report], "all": [cmd_stats, cmd_derive, cmd_run, cmd_report]}
    if cmd not in steps:
        raise SystemExit(__doc__)
    for fn in steps[cmd]:
        fn()
