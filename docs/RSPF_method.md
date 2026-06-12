# RS-PF Smoother — Regime-Switching Particle Filter

A second, independent HRV smoothing method for benchmarking against the existing
`particles` SMC + FFBS smoother (`02_run_filters.py`). It is a NumPy port of the
regime-switching bootstrap particle filter from
[WickhamLi/RS-DBPF](https://github.com/WickhamLi/RS-DBPF)
([Differentiable Bootstrap Particle Filters for Regime-Switching Models](https://arxiv.org/abs/2302.10319)),
rewritten to obey the project's H-Framework axioms.

The existing particle-filter pipeline is **not modified**. This method writes to
a parallel directory so both can be annotated and compared.

```
data/processed/<pid>_processed.csv   →   data/smoothed_rspf/<pid>_rspf.csv
```

## Files

| File | Role |
|---|---|
| `src/rspf_smoother.py` | Core vectorized RS-PF. Public API `smooth_dataframe(df, ...)`; accepts a DataFrame with `patient_id`, `timestamp`, `hrv_value`. Run directly (`python src/rspf_smoother.py`) for the self-test. |
| `src/02b_run_rspf.py` | Batch runner mirroring `02_run_filters.py` conventions (ProcessPool, per-patient CSVs, log). `python src/02b_run_rspf.py`. |

## What was kept from RS-DBPF

The scientific core: a latent regime `m_t ∈ {0..K-1}` that switches via a sticky
Markov transition matrix, and a continuous latent state `s_t` whose dynamics
depend on the active regime. This is exactly what lets one model represent a
baseline that jumps between physiological states (awake vs asleep). The forward
bootstrap filter and ESS-triggered systematic resampling are kept.

The **differentiable / neural-network** machinery was dropped on purpose: it
requires labeled ground-truth trajectories (`s_test`) to train its proposal and
measurement networks, which wearable HRV data does not have. The classical RS-PF
keeps the regime-switching idea while being fully self-contained.

## Where the original RS-DBPF violated the H-Framework → how the port fixes it

| Axiom | RS-DBPF as written | Fix in this port |
|---|---|---|
| **H1** Bounded positive (HRV > 0 ms) | State is an unbounded Gaussian AR process; observation `C·√\|s\|+D` is defined for negative `s` and the posterior mean can sit at/below 0. | The observable is `lo + (hi−lo)·sigmoid(level + circadian)`, where `[lo, hi]` is the patient's exact observed `[min, max]`. The sigmoid confines **every** output (smoothed AND trend) strictly inside `(lo, hi)` — guaranteeing positivity and that nothing exceeds the patient's own range. (An earlier log-space version only bounded below and let the unidentified level/circadian split push the trend to 271 ms on a [22,129] patient — fixed by the bounded map.) |
| **H2 / H4** Multimodal, drifting baseline | Already present (regime switching) — the reason this base method was chosen. | Each regime `k` has its own baseline `μ_k` in logit space (set from per-patient quantiles). The within-regime level is a **slow, stiff** AR(1) (τ≈720 min, small shock) so smoothing comes from the stiff level; the big awake↔asleep jumps come from **sticky** Markov regime switches (self-transition 0.985). This is what makes it denoise instead of mimic the raw. |
| **H5** Circadian ~24h rhythm | No time-of-day term at all. | A shared circadian regressor `β_c·cos(ωt)+β_s·sin(ωt)` (`ω = 2π/24h`, from the wall-clock timestamp) is added to every particle's log-mean; amplitude is fit causally online. |
| **H7** Irregular sampling / jitter | Assumes a fixed unit time step. | Every transition is **dt-aware**: AR retention `φ = exp(−Δt/τ)` and process variance both scale with the actual elapsed minutes, so 12-min jitter and a 120-min within-chunk gap are handled differently and correctly. |
| **H8** Massive missingness / segmentation (≥180 min ⇒ new chunk) | Filters one continuous trajectory end-to-end; would smooth across a 4-hour void. | `time_diff` between consecutive **readings** is computed, `≥ 180 min` flagged, a unique `chunk_id` assigned, and smoothing grouped by `[patient_id, chunk_id]`. The particle cloud, weights, and regime distribution are **fully re-initialised at every boundary**; the first row of a chunk is `t=0`. |
| **H9** Causal / online (no lookahead) | Forward filter is causal, *but* the headline results train by backprop over whole trajectories, and this project's companion smoother uses FFBS backward sampling (looks ahead → violates H9). | Output is strictly the **filtered posterior mean** `E[s_t \| o_{1:t}]` — past and present only. No backward pass, no centering, no future window. Verified: truncating the series after `t` leaves every estimate up to `t` byte-identical. |
| **H10** Patient variability | Global / pooled coefficients. | Regime baselines, process/observation noise scales, and the latent prior are all derived **per patient** from that patient's own observed log-HRV. Nothing is shared across patients. |

### A note on H8 and the preprocessed files

`00_preprocess_raw.py` inserts NaN-HRV filler rows every 10 min across gaps
`> 180 min`. Those fillers make consecutive *row* timestamps look ~10 min apart,
which would **hide the real voids** from a naive row-adjacency chunker (on patient
0010, that finds 4 chunks instead of the true 146). The H8 rule is defined on
consecutive *readings*, so the runner drops the fillers, computes `time_diff`
between real readings, segments there, then merges the smoothed values back onto
the full grid for row-alignment parity with `data/smoothed/`.

## Output schema

Identical to `data/smoothed/<pid>_smoothed.csv`, plus `chunk_id`:

```
createdTime, hrvValue, minute_diff, smoothed_hrv, true_trend_level, gap_flag, chunk_id
```

* `smoothed_hrv` — RS-PF causal filtered estimate (full signal, `> 0`)
* `true_trend_level` — circadian-removed baseline (the trend level, `> 0`)
* `chunk_id` — H8 segment id (resets at ≥180-min reading gaps)

Because `03_annotate.py` keys on `createdTime` + `smoothed_hrv` (and prefers
`true_trend_level` when present), you can benchmark CPD on this method by pointing
its `INPUT_DIR` at `data/smoothed_rspf`.

## Configuration

Both live as one-line constants in `src/02b_run_rspf.py` (and `RSPFConfig`):

| Constant | Meaning | Default |
|---|---|---|
| `GLOBAL_K` | number of latent regimes (H2/H4): 2 = awake/asleep, 3 = asleep/rest/active | `3` |
| `GLOBAL_N_PARTICLES` | particles per chunk (higher = smoother, linearly slower) | `500` |
| `GAP_THRESHOLD_MIN` | H8 segmentation boundary | `180` |
| `RSPFConfig.stickiness` | Markov self-transition probability (regime persistence) | `0.97` |
| `RSPFConfig.reversion_tau_min` | AR(1) baseline reversion timescale (minutes) | `240` |

## Verification (run `python src/rspf_smoother.py`)

* **H1 bounds** — every smoothed and trend value lies strictly inside the patient's `[min, max]`. On 0010 (raw [22, 129]): smoothed [23.6, 128.7], trend [26.1, 127.5], **zero** out-of-bounds points.
* **Denoising (not mimicry)** — on 0010 the smoother removes real noise: raw std 35.6 → smoothed std 24.1 (≈26 ms RMSE removed), rather than tracing every reading.
* **H8 segmentation** — observed chunk count matches the real ≥180-min gap count (0010: 146); max within-chunk reading gap is `< 180 min`, proving nothing smooths across a void.
* **H9 causality** — with calibration held fixed, smoothing a prefix vs the full series gives `max |Δ| = 0.00e+00` on shared rows.
