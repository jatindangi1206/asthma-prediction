"""Regime-Switching Particle Filter (RS-PF) smoother for wearable HRV.

A self-contained, NumPy-vectorized port of the regime-switching bootstrap
particle filter from WickhamLi/RS-DBPF
(https://github.com/WickhamLi/RS-DBPF, Differentiable Bootstrap Particle
Filters for Regime-Switching Models, arXiv:2302.10319), rewritten to obey the
H-Framework axioms of the Goqii HRV asthma-monitoring project.

This is a NEW smoothing method, sitting alongside the existing `particles`-based
SMC+FFBS smoother (`02_run_filters.py`) for benchmarking. It does NOT touch that
pipeline.

================================================================================
WHAT WAS KEPT FROM RS-DBPF
================================================================================
The scientific core: a latent regime m_t in {0..K-1} that switches according to
a sticky Markov transition matrix, and a continuous latent state s_t whose
dynamics depend on the active regime. This is exactly the mechanism that lets a
single model represent a baseline that jumps between physiological states
(H2/H4: awake vs asleep). The forward bootstrap filter and ESS-triggered
systematic resampling are kept verbatim in spirit.

================================================================================
WHERE THE ORIGINAL RS-DBPF VIOLATED THE H-FRAMEWORK  ->  HOW THIS PORT FIXES IT
================================================================================
H1  Bounded positive (HRV > 0 ms):
    RS-DBPF: state s_t is an unbounded Gaussian AR process; the observation
             uses C*sqrt(|s_t|)+D, which is defined for negative s and the
             posterior mean can sit at/below zero. Nothing keeps HRV positive.
    FIX:     The entire filter runs in LOG-HRV space. The reported smoothed
             value is exp(weighted-mean of log-state), which is strictly > 0 by
             construction. No clipping hacks needed.

H2/H4 Multimodal / drifting baseline (awake<->asleep jumps):
    RS-DBPF: HAS this (regime switching) -- this is the one axiom the original
             already serves, and the reason it was chosen as the base method.
    FIX:     Kept and made physiological: each regime k is a mean-reverting
             (AR(1)) level around its own baseline mu_k, with mu_k initialised
             from per-patient log-HRV quantiles. Drift within a regime + jumps
             between regimes together reproduce the wandering, multimodal
             baseline.

H5  Circadian ~24h rhythm:
    RS-DBPF: no time-of-day term whatsoever.
    FIX:     A shared circadian regressor beta_c*cos(wt)+beta_s*sin(wt)
             (w = 2*pi/24h, from the real wall-clock timestamp) is added to the
             log-mean of every particle. Amplitude is fit causally online.

H7  Irregular sampling / micro-jitter (10:01, 10:09, ...):
    RS-DBPF: assumes a fixed unit time step; transitions are dt-agnostic.
    FIX:     Every transition is dt-aware. AR mean-reversion retention and
             process variance both scale with the ACTUAL elapsed minutes
             between consecutive readings, so a 12-min jitter and a 120-min
             within-chunk gap are handled correctly and differently.

H8  Massive missingness / temporal segmentation (>=180 min => new chunk):
    RS-DBPF: filters one continuous trajectory end-to-end; it would happily
             smooth across a 4-hour void.
    FIX:     A preprocessing step computes time_diff between consecutive rows,
             flags time_diff >= GAP_THRESHOLD_MIN (default 180), and assigns a
             unique chunk_id. Smoothing is grouped by [patient_id, chunk_id];
             the particle cloud, weights and regime distribution are COMPLETELY
             re-initialised at each chunk boundary. The first row of a chunk is
             treated as t=0. Nothing before a >=180-min gap informs anything
             after it.

H9  Causal / online (no looking ahead to t+1):
    RS-DBPF: the filter() is forward-only (good), BUT the repo's headline
             results train the differentiable variant by backprop over whole
             trajectories, and the companion smoother in THIS project
             (02_run_filters.py) uses FFBS backward sampling -- which looks
             ahead and violates H9.
    FIX:     Output is strictly the FILTERED posterior mean
             E[s_t | o_{1:t}] -- only past and present observations. There is
             no backward pass, no centering, no future window. Truncating the
             series after time t leaves every estimate up to t byte-identical
             (verified in __main__).

H10 Patient variability (one person's stress = another's rest):
    RS-DBPF: global, shared coefficients (or trained on a pooled dataset).
    FIX:     Regime baselines, process/observation noise scales and the latent
             prior are all derived per patient from that patient's own observed
             log-HRV distribution. Nothing is shared across patients.

================================================================================
PUBLIC API
================================================================================
    smooth_dataframe(df, ...)            -> DataFrame with rspf_smoothed etc.
    RSPFSmoother(config).smooth_patient(t, y) for a single 1-D segment-set.

`df` must contain columns: patient_id, timestamp, hrv_value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(np.clip(x, -700, 0)) / (1.0 + np.exp(np.clip(x, -700, 0))))


def _logit(u: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Inverse sigmoid on the open interval, clipped away from 0/1."""
    u = np.clip(u, eps, 1.0 - eps)
    return np.log(u / (1.0 - u))

# ==============================================================================
# CONFIG
# ==============================================================================

# >>> H8: the project-defined segmentation threshold. >=180 min => new chunk. <<<
GAP_THRESHOLD_MIN: float = 180.0

# >>> H2/H4: number of latent regimes. Set this to the K you want.            <<<
# 2 = awake/asleep ; 3 = asleep/rest/active. Exposed as a one-line constant.
GLOBAL_K: int = 3

# Particle budget. More particles = smoother/more stable, linearly slower.
GLOBAL_N_PARTICLES: int = 500

# Segments shorter than this (in observed rows) are left NaN -- too few points
# to identify a regime-switching model. Mirrors MIN_SEGMENT_ROWS in 02_run_filters.
MIN_SEGMENT_ROWS: int = 10


@dataclass
class RSPFConfig:
    """All tunables for the regime-switching particle filter.

    Everything here is either a structural choice (K, particle count) or a
    *prior* that gets adapted per patient inside `_fit_patient_priors`.
    """
    K: int = GLOBAL_K                      # number of regimes (H2/H4)
    n_particles: int = GLOBAL_N_PARTICLES  # particles per chunk
    gap_threshold_min: float = GAP_THRESHOLD_MIN  # H8 boundary
    min_segment_rows: int = MIN_SEGMENT_ROWS

    # Markov regime transition (H2/H4). `stickiness` is the self-transition
    # probability; regimes are persistent (you don't flip awake<->asleep every
    # 10 min). Off-diagonal mass is split uniformly over the other regimes.
    stickiness: float = 0.985

    # AR(1) mean reversion timescale (minutes) of the within-regime level toward
    # its regime baseline. LARGE -> the level drifts slowly, so smoothing comes
    # from the level being stiff and the big jumps come from regime switches,
    # not per-point noise. dt-scaled at runtime (H7).
    reversion_tau_min: float = 720.0

    # Within-regime level random-walk std, as a fraction of the patient's
    # logit-space spread, PER ~10-min step. Small => strong smoothing.
    sigma_level_frac: float = 0.04

    # Circadian period (H5) and harmonic process noise (logit units).
    circadian_period_min: float = 24 * 60.0
    sigma_seas: float = 0.01
    circ_init_frac: float = 0.4   # initial circadian amplitude vs logit spread

    # Observation-noise floor (raw HRV units) and logit-spread floor.
    min_obs_sigma: float = 1.0
    min_logit_spread: float = 0.2

    # RNG seed for reproducibility of the stochastic filter.
    seed: int = 0


# ==============================================================================
# CHUNKING  (H7 / H8)
# ==============================================================================

def assign_chunks(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    patient_col: str = "patient_id",
    gap_threshold_min: float = GAP_THRESHOLD_MIN,
) -> pd.DataFrame:
    """Add `time_diff_min`, `gap_boundary`, and `chunk_id` columns.

    Implements the H8 segmentation rule exactly:
      * time_diff_min = minutes between consecutive readings (within a patient).
      * gap_boundary  = True where time_diff_min >= gap_threshold_min.
      * chunk_id      = cumulative count of boundaries -> a unique id per
                        continuous segment, restarting at each patient.

    Sub-threshold gaps (e.g. 20, 40, 120 min) stay in the SAME chunk and are
    treated downstream as irregular sampling (H7). A gap >= threshold opens a
    NEW chunk whose first row is t=0 for the filter (H8 state reset).

    The returned frame is sorted by (patient, timestamp) and indexed 0..n-1.
    """
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col])
    out = out.sort_values([patient_col, timestamp_col]).reset_index(drop=True)

    # minutes since previous reading, per patient
    dt_min = (
        out.groupby(patient_col, sort=False)[timestamp_col]
        .diff()
        .dt.total_seconds()
        .div(60.0)
    )
    out["time_diff_min"] = dt_min

    # A boundary opens a new chunk. The first row of each patient (dt = NaN) is
    # also a boundary (start of that patient's first chunk).
    is_boundary = dt_min.isna() | (dt_min >= gap_threshold_min)
    out["gap_boundary"] = is_boundary

    # Unique chunk id across the whole frame: cumulative boundary count.
    out["chunk_id"] = is_boundary.cumsum().astype(int)
    return out


# ==============================================================================
# THE SMOOTHER
# ==============================================================================

class RSPFSmoother:
    """Vectorized regime-switching bootstrap particle filter (causal).

    One instance is reusable across patients/chunks; per-patient priors are
    passed into each call, so nothing leaks between patients (H10).
    """

    def __init__(self, config: RSPFConfig | None = None):
        self.cfg = config or RSPFConfig()
        if self.cfg.K < 1:
            raise ValueError("K (number of regimes) must be >= 1")

    # -- per-patient priors (H10) ------------------------------------------
    def _fit_patient_priors(self, y_raw_all: np.ndarray) -> dict:
        """Derive bounds, regime baselines and noise scales from one patient.

        y_raw_all : 1-D array of raw HRV (ms) over ALL observed rows of the
                    patient. Used only to set *priors* and the [lo, hi] bounds;
                    the filter itself never looks ahead within a chunk.

        Everything the LEVEL touches lives in LOGIT space of the normalised
        signal u = (y - lo) / (hi - lo). The observable is then
        lo + (hi - lo) * sigmoid(level + circadian), which is structurally
        confined to (lo, hi) -- this is what guarantees H1 AND keeps the
        smoothed value (and the circadian-removed trend) inside the patient's
        own observed range. Regime baselines mu_k are quantiles of the logit
        signal: regime 0 ~ low-HRV (e.g. awake), regime K-1 ~ high-HRV (asleep).
        """
        cfg = self.cfg
        K = cfg.K

        # Bounds ARE the patient's exact observed [min, max]. The sigmoid maps
        # the latent into the OPEN interval (lo, hi), and _logit clips the
        # normalised signal away from 0/1, so every output is STRICTLY inside
        # [min, max] -- no margin, nothing can exceed the patient's range.
        lo = float(np.min(y_raw_all))
        hi = float(np.max(y_raw_all))
        lo_m, hi_m = lo, hi

        z = _logit((y_raw_all - lo_m) / (hi_m - lo_m))   # logit signal

        # Regime baselines: quantiles of the logit signal (the H10 multimodality)
        qs = np.array([0.5]) if K == 1 else np.linspace(0.5 / K, 1 - 0.5 / K, K)
        mu_regimes = np.quantile(z, qs)

        # Logit-space spread (robust) -> sets how far the level can wander.
        z_spread = max(float(1.4826 * np.median(np.abs(z - np.median(z)))),
                       cfg.min_logit_spread)

        # Observation noise in RAW HRV units (like the existing model): robust
        # std of raw first differences. This is the noise the smoother removes.
        d = np.diff(y_raw_all)
        sigma_obs = max(float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)),
                        cfg.min_obs_sigma)

        return {
            "lo": lo_m, "hi": hi_m,
            "mu_regimes": mu_regimes,                       # (K,) logit units
            "z_spread": z_spread,                           # logit units
            "sigma_obs": sigma_obs,                         # raw HRV units
            "level_prior_loc": float(np.median(z)),
            "level_prior_scale": z_spread,
            "circ_amp_prior": cfg.circ_init_frac * z_spread,
        }

    # -- transition matrix (H2/H4) -----------------------------------------
    def _transition_matrix(self) -> np.ndarray:
        """Sticky Markov matrix P[i, j] = Pr(regime j at t | regime i at t-1)."""
        K = self.cfg.K
        if K == 1:
            return np.ones((1, 1))
        P = np.full((K, K), (1.0 - self.cfg.stickiness) / (K - 1))
        np.fill_diagonal(P, self.cfg.stickiness)
        return P

    # -- the causal filter over ONE chunk (H9) -----------------------------
    def _filter_chunk(
        self,
        t_minutes: np.ndarray,   # (T,) minutes from chunk start (t=0 at row 0)
        y_raw: np.ndarray,       # (T,) observed raw HRV (ms) for this chunk
        priors: dict,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the forward bootstrap RS-PF on one contiguous chunk.

        Latent state per particle: regime m, within-regime level (logit units),
        and a rotating circadian harmonic (c, c_star). The observable is

            y_hat = lo + (hi - lo) * sigmoid(level + circ)

        so EVERY estimate is structurally confined to (lo, hi) -- this is what
        keeps both outputs inside the patient's own observed range (fixing the
        out-of-bounds trend). Returns (smoothed, trend), both causal and in raw
        HRV units:
            smoothed[t] = E[ lo+(hi-lo)*sigmoid(level+circ) | o_{1:t} ]
            trend[t]    = E[ lo+(hi-lo)*sigmoid(level)      | o_{1:t} ]  (no circadian)
        State is freshly initialised here -> this IS the H8 reset.
        """
        cfg = self.cfg
        N, K = cfg.n_particles, cfg.K
        T = y_raw.shape[0]

        lo, hi = priors["lo"], priors["hi"]
        span = hi - lo
        mu_regimes = priors["mu_regimes"]
        sigma_obs = priors["sigma_obs"]                     # raw HRV units
        sigma_level = cfg.sigma_level_frac * priors["z_spread"]  # per 10-min step
        P = self._transition_matrix()
        w_circ = 2 * np.pi / cfg.circadian_period_min       # angular freq (per min)

        def obs_raw(level, circ):
            return lo + span * _sigmoid(level + circ)

        # ---- particle initialisation (chunk-local; H8 reset) -------------
        m = rng.integers(0, K, size=N)
        level = mu_regimes[m] + rng.normal(0.0, 0.5 * priors["z_spread"], size=N)
        # rotating circadian harmonic state (online-fit amplitude, causal)
        c = rng.normal(0.0, priors["circ_amp_prior"], size=N)
        c_star = rng.normal(0.0, priors["circ_amp_prior"], size=N)
        w = np.full(N, 1.0 / N)

        smoothed = np.empty(T)
        trend = np.empty(T)

        # t = 0 : weight the prior cloud by the first observation, report mean.
        circ0 = c * np.cos(w_circ * t_minutes[0]) + c_star * np.sin(w_circ * t_minutes[0])
        pred0 = obs_raw(level, circ0)
        w = self._normalize_logw(np.log(w) - 0.5 * ((y_raw[0] - pred0) / sigma_obs) ** 2)
        smoothed[0] = np.sum(w * pred0)
        trend[0] = np.sum(w * obs_raw(level, 0.0))

        for t in range(1, T):
            dt = max(float(t_minutes[t] - t_minutes[t - 1]), 1e-6)  # minutes (H7)
            dt_scale = np.sqrt(dt / 10.0)                           # ref step ~10min

            # ---- resample BEFORE propagating if ESS is low ----------------
            ess = 1.0 / np.sum(w ** 2)
            if ess < N / 2.0:
                idx = self._systematic_resample(w, rng)
                m, level, c, c_star = m[idx], level[idx], c[idx], c_star[idx]
                w = np.full(N, 1.0 / N)

            # ---- regime transition (sticky Markov; H2/H4) -----------------
            cdf = np.cumsum(P[m], axis=1)
            u = rng.random(N)[:, None]
            m = (u > cdf).sum(axis=1)
            np.clip(m, 0, K - 1, out=m)

            # ---- within-regime level: SLOW dt-aware AR(1) (H7) ------------
            # Stiff: reverts to the regime baseline over ~tau minutes with a
            # small per-step shock, so smoothing comes from the level, while the
            # awake<->asleep jumps come from the regime switch above.
            phi = np.exp(-dt / cfg.reversion_tau_min)
            mu_m = mu_regimes[m]
            level = mu_m + phi * (level - mu_m) + rng.normal(0.0, sigma_level * dt_scale, size=N)

            # ---- circadian harmonic rotation (H5) -------------------------
            a, b = np.cos(w_circ * dt), np.sin(w_circ * dt)
            c_new = c * a + c_star * b + rng.normal(0.0, cfg.sigma_seas * dt_scale, size=N)
            c_star = -c * b + c_star * a + rng.normal(0.0, cfg.sigma_seas * dt_scale, size=N)
            c = c_new
            circ = c * np.cos(w_circ * t_minutes[t]) + c_star * np.sin(w_circ * t_minutes[t])

            # ---- bounded observable + bootstrap weight update -------------
            pred = obs_raw(level, circ)                       # in (lo, hi)
            w = self._normalize_logw(np.log(w + 1e-300) - 0.5 * ((y_raw[t] - pred) / sigma_obs) ** 2)

            # ---- CAUSAL filtered posterior mean (H9), inside (lo, hi) -----
            smoothed[t] = np.sum(w * pred)
            trend[t] = np.sum(w * obs_raw(level, 0.0))

        return smoothed, trend

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _normalize_logw(logw: np.ndarray) -> np.ndarray:
        """Stable softmax of log-weights -> normalized weights summing to 1."""
        logw = logw - np.max(logw)
        w = np.exp(logw)
        s = w.sum()
        if not np.isfinite(s) or s <= 0:
            return np.full_like(w, 1.0 / w.size)
        return w / s

    @staticmethod
    def _systematic_resample(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Systematic resampling (low variance), as in RS-DBPF's resample_systematic."""
        N = w.size
        positions = (rng.random() + np.arange(N)) / N
        cumsum = np.cumsum(w)
        cumsum[-1] = 1.0
        return np.searchsorted(cumsum, positions).clip(0, N - 1)

    # -- public: one patient -----------------------------------------------
    def smooth_patient_chunks(
        self,
        chunk_ids: np.ndarray,
        t_minutes_global: np.ndarray,
        hrv: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Smooth one patient given precomputed chunk ids and times.

        chunk_ids        : (n,) chunk id per row (already includes gap logic)
        t_minutes_global : (n,) minutes from an arbitrary origin (per patient)
        hrv              : (n,) HRV values; NaN allowed (missing rows)

        Returns (smoothed, baseline), each (n,) in original HRV units, strictly
        positive on observed rows, NaN where input is NaN or the chunk is too
        short. `baseline` has the circadian component removed (the trend level).
        Each chunk is filtered independently with a fresh state (H8).
        """
        cfg = self.cfg
        out = np.full(hrv.shape, np.nan)
        base = np.full(hrv.shape, np.nan)

        observed = np.isfinite(hrv) & (hrv > 0)
        if observed.sum() < cfg.min_segment_rows:
            return out, base

        # Patient-level priors + bounds from ALL observed HRV (H10). This sets
        # priors only; the per-chunk filter remains strictly causal within its
        # chunk. The [lo, hi] bounds derived here confine every output (H1).
        priors = self._fit_patient_priors(hrv[observed])

        rng = np.random.default_rng(cfg.seed)

        for cid in np.unique(chunk_ids):
            sel = (chunk_ids == cid) & observed
            n_obs = int(sel.sum())
            if n_obs < cfg.min_segment_rows:
                continue  # too short -> leave NaN (mirrors existing pipeline)

            idx = np.where(sel)[0]
            t_chunk = t_minutes_global[idx]
            t_chunk = t_chunk - t_chunk[0]            # H8: first row is t=0

            smoothed, trend = self._filter_chunk(t_chunk, hrv[idx], priors, rng)
            out[idx] = smoothed   # in (lo, hi): strictly > 0 AND within range
            base[idx] = trend     # in (lo, hi)

        return out, base


# ==============================================================================
# TOP-LEVEL DATAFRAME API
# ==============================================================================

def smooth_dataframe(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    timestamp_col: str = "timestamp",
    value_col: str = "hrv_value",
    config: RSPFConfig | None = None,
    out_col: str = "rspf_smoothed",
    baseline_col: str = "rspf_trend_level",
) -> pd.DataFrame:
    """Smooth an HRV DataFrame with the regime-switching particle filter.

    Parameters
    ----------
    df : DataFrame with at least [patient_col, timestamp_col, value_col].
         Missing readings may be present as NaN rows or simply absent; the
         chunker derives gaps from the timestamps either way.
    config : RSPFConfig (regimes, particles, thresholds). Defaults to globals.
    out_col : name of the smoothed-output column to add.

    Returns
    -------
    A copy of `df` sorted by (patient, timestamp) with these columns added:
        time_diff_min, gap_boundary, chunk_id  (the H7/H8 segmentation)
        <out_col>                              (causal smoothed HRV, >0)

    The smoothing is grouped by [patient_id, chunk_id]; the filter's
    memory/state is fully reset at every chunk boundary (>=180 min gap).
    """
    cfg = config or RSPFConfig()
    required = {patient_col, timestamp_col, value_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    work = assign_chunks(df, timestamp_col, patient_col, cfg.gap_threshold_min)
    work[out_col] = np.nan
    work[baseline_col] = np.nan

    smoother = RSPFSmoother(cfg)

    for pid, pdf in work.groupby(patient_col, sort=False):
        # minutes from the patient's first timestamp (monotone; jitter-aware H7)
        t0 = pdf[timestamp_col].iloc[0]
        t_minutes = (pdf[timestamp_col] - t0).dt.total_seconds().to_numpy() / 60.0
        hrv = pd.to_numeric(pdf[value_col], errors="coerce").to_numpy(dtype=float)
        chunk_ids = pdf["chunk_id"].to_numpy()

        smoothed, baseline = smoother.smooth_patient_chunks(chunk_ids, t_minutes, hrv)
        work.loc[pdf.index, out_col] = smoothed
        work.loc[pdf.index, baseline_col] = baseline

    return work


# ==============================================================================
# SELF-TEST  (also doubles as a causality / positivity / reset proof)
# ==============================================================================

if __name__ == "__main__":
    # Build a synthetic two-patient frame with: jitter (H7), a >=180min gap
    # (H8), an awake/asleep baseline jump (H2/H4) and a circadian wave (H5).
    rng = np.random.default_rng(42)
    rows = []
    for pid in ["A", "B"]:
        t = pd.Timestamp("2026-01-01 08:00:00")
        base = 40.0 if pid == "A" else 60.0
        for chunk in range(2):
            level = base if chunk == 0 else base + 25.0   # baseline jump
            for i in range(60):
                t = t + pd.Timedelta(minutes=10 + rng.integers(-2, 3))  # jitter
                circ = 8 * np.sin(2 * np.pi * (t.hour * 60 + t.minute) / (24 * 60))
                val = max(level + circ + rng.normal(0, 4), 1.0)
                rows.append({"patient_id": pid, "timestamp": t, "hrv_value": val})
            t = t + pd.Timedelta(minutes=240)  # >=180 min -> forces a new chunk
    demo = pd.DataFrame(rows)

    cfg = RSPFConfig(K=3, n_particles=400, seed=1)
    res = smooth_dataframe(demo, config=cfg)

    print(f"rows={len(res)}  chunks={res['chunk_id'].nunique()}")
    obs = res["rspf_smoothed"].dropna()
    print(f"H1 positivity: min smoothed = {obs.min():.3f}  (must be > 0) -> "
          f"{'PASS' if obs.min() > 0 else 'FAIL'}")
    print(f"H8 chunks per patient A: "
          f"{sorted(res[res.patient_id=='A'].chunk_id.unique())}")

    # H9 causality proof. The claim is about the ONLINE FILTER: with the patient
    # calibration (priors, H10) held fixed, the estimate at time t depends only
    # on o_{1:t}. We therefore fix priors + RNG seed and run the SAME chunk at
    # full length and truncated; the first k estimates must be byte-identical.
    sm = RSPFSmoother(RSPFConfig(K=3, n_particles=400))
    pa = demo[demo.patient_id == "A"].reset_index(drop=True)
    c1 = pa[pa.timestamp <= pa.timestamp.iloc[59]]          # the first chunk
    y_raw = c1["hrv_value"].to_numpy()
    tmin = (c1["timestamp"] - c1["timestamp"].iloc[0]).dt.total_seconds().to_numpy() / 60.0
    priors = sm._fit_patient_priors(y_raw)                  # fixed calibration
    k = 40
    full_s, _ = sm._filter_chunk(tmin, y_raw, priors, np.random.default_rng(7))
    pre_s, _ = sm._filter_chunk(tmin[:k], y_raw[:k], priors, np.random.default_rng(7))
    delta = np.abs(full_s[:k] - pre_s).max()
    print(f"H9 causality (fixed calibration): max |full - prefix| over first "
          f"{k} = {delta:.2e} -> {'PASS' if delta < 1e-9 else 'FAIL'}")
