# Smoothing — round 1

*Status as of 3 Nov 2026. This is the first round of method selection. Nothing here is
final: the candidates and their settings will be compared again once the prediction
models are in place, and adjusted to what the models actually need.*

## 1. What smoothing is for

A wearable reports HRV (SDNN, one reading about every 10 minutes) and HR (one reading a
minute). Both are noisy, irregular and full of gaps. A **smoother** estimates the
patient's underlying level at each reading — the physiological baseline — so that each
channel yields three signals:

| signal | meaning | used for |
|---|---|---|
| raw | what the watch reported | always a model feature |
| smoothed | the estimated baseline | a feature; change detection in path 4 |
| residual = raw − smoothed | the departure from the baseline right now | a feature; change detection in path 5 (the main hypothesis) |

All three always reach the model. What a path decides is which one the change-point
detectors read (see `docs/change_point_detection_round1.md`). So the smoother matters
twice: it shapes the model's features, and it decides what counts as "unusual" in the
residual.

**What the smoother's memory implies.** A smoother that adapts within hours removes
anything slower than that from the residual. Measured: anrewma's normalisation forgets
in about 3.3 h at default settings, gammadglm's level discount in about 17 h. A
prodrome that builds over several days is therefore absorbed into the baseline before a
residual-based detector can see it. That is why the CPD round added a separate
nightly-level detector.

## 2. The data the smoothers face

Measured on this cohort (pass1 export; 187 patients with HRV, 205 with HR):

| property | HRV | HR |
|---|---|---|
| readings per worn day (median) | 47 | 786 |
| nominal cadence | ~10 min, irregular (13% of steps > 20 min) | 1 min |
| gaps ≥ 180 min per worn day | 1.25 | 0.6 |
| noise share of the raw variance (variogram nugget) | ~30% | — |
| level correlation time τ (variogram) | ~140–150 min (IQR 75–277) | ~7–9 min (residual) |
| device ceiling | 129; 2.7% of readings exactly 129, **29% in 120–129** | — |

The variogram model is `γ(h) = R + S·(1 − e^(−h/τ))`: R is the measurement-noise
variance, S the variance of the wandering level, τ how long the level stays
correlated (`wearable/cohort_stats.py`).

Decision recorded: the 120–129 HRV readings are kept as ordinary data everywhere. Their
pile-up (5.5× the density just below 120, more by day and at high HR) is documented, but
no artifact interpretation is applied.

## 3. The ten candidate smoothers

Each is one file in `wearable/smoothing/`, discovered automatically. All run per
patient, restart at gaps ≥ 180 min (H8) and use real elapsed time (H7).

| method | family | idea in one line |
|---|---|---|
| anrewma | adaptive EWMA | causal rolling normalisation + an outlier-weighted EWMA |
| ctimm | interacting multiple model | continuous-time switching between dynamic modes |
| gammadglm | dynamic GLM | Gamma observation model with a discounted level + daily harmonics |
| gpssm | Gaussian process (state space) | GP prior on the level, filtered forward |
| hrkf | robust Kalman | Huber-weighted Kalman filter with level and seasonal |
| kim | Markov switching | Kim filter over regimes |
| krlst | kernel recursive least squares | budgeted kernel tracker with forgetting |
| logmhw | Holt–Winters | log-scale multiplicative Holt–Winters |
| ossa | singular spectrum analysis | online SSA on a sliding window |
| rspf | particle filter | regime-switching bootstrap particle filter (`docs/RSPF_method.md`) |

## 4. What round 1 did

`scripts/smoothing_study.py stats | derive | run | report`:

1. **Characterise each cohort** (AIIMS adults + children, COPD, combined): cadence,
   gaps, variogram (R, S, τ), circadian amplitude, volatility, activity state.
2. **Derive every smoother's settings from those statistics**, with a written reason for
   each knob (`data/reports/smoothing/settings.csv`). Example: anrewma's `norm_alpha`
   becomes `1 − exp(−dt/τ_day)` with τ_day = 1021 min, from the day-to-day correlation
   of daily medians.
3. **Run all ten twice** — at their default settings and at the derived ones — on each
   cohort.
4. **Score each against the raw signal** with six signal metrics (no labels, no model):

| metric | what it asks | better |
|---|---|---|
| correlation | does the baseline follow the raw signal? | higher |
| variance_reduction | how much variance was removed? | (closer to ideal) |
| roughness_reduction | how much minute-to-minute jitter was removed? | (closer to ideal) |
| peak_preservation | were real bursts kept while jitter was removed? | higher |
| residual_std | spread of raw − smoothed | (closer to ideal) |
| mean_error_pct | systematic offset from the raw signal | lower |

5. **Compare with an "ideal" reference**: the scores a perfect causal filter would get
   on simulated data with the cohort's own variogram (level + white noise). Removing
   *all* variation is not the goal: with ~30% of the variance being measurement noise,
   the ideal removes about 41% of the variance, keeps a correlation of about 0.92, and
   preserves peaks at about 0.42.

## 5. Results — HRV, combined cohort (187 patients)

| method | settings | corr | var. red. % | rough. red. % | peak pres. | resid. sd | mean err % |
|---|---|---|---|---|---|---|---|
| *ideal reference* | | *0.92* | *41.5* | *66.4* | *0.42* | *13.6* | *0.0* |
| anrewma | default | 0.54 | 50.6 | 80.2 | 0.23 | 28.1 | 3.1 |
| anrewma | derived | 0.68 | 50.3 | 71.2 | 0.34 | 24.4 | 2.3 |
| ctimm | default | 0.75 | 59.7 | 73.0 | 0.29 | 21.7 | 4.1 |
| ctimm | derived | 0.93 | 37.2 | 46.1 | 0.59 | 12.6 | 2.8 |
| gammadglm | default | 0.72 | 47.5 | 67.8 | 0.38 | 22.6 | 2.8 |
| gammadglm | derived | 0.78 | 23.2 | 45.8 | 0.62 | 20.8 | 1.1 |
| gpssm | default | 0.17 | 97.5 | 95.5 | 0.00 | 32.1 | 9.3 |
| gpssm | derived | 0.98 | 48.3 | 34.2 | 0.67 | 10.4 | 3.9 |
| hrkf | default | 0.62 | 41.9 | 71.8 | 0.33 | 25.7 | 1.5 |
| hrkf | derived | 0.88 | 37.3 | 48.7 | 0.57 | 16.0 | 1.9 |
| kim | default | 0.73 | 68.7 | 76.5 | 0.27 | 22.7 | 4.0 |
| kim | derived | 0.87 | 49.7 | 56.6 | 0.48 | 16.9 | 5.0 |
| krlst | default | 0.70 | 90.2 | 87.8 | 0.15 | 26.8 | 8.2 |
| krlst | derived | 0.77 | 59.1 | 70.7 | 0.34 | 21.2 | 4.8 |
| logmhw | default | 0.62 | 51.7 | 80.4 | 0.24 | 25.9 | 4.3 |
| logmhw | derived | 0.60 | 60.6 | 81.8 | 0.22 | 26.6 | 5.2 |
| ossa | default | 0.64 | 38.7 | 48.4 | 0.45 | 25.6 | 3.5 |
| ossa | derived | 0.86 | 15.0 | 17.2 | 0.91 | 17.0 | 1.7 |
| rspf | default | 0.64 | 65.1 | 79.4 | 0.26 | 25.0 | 1.3 |
| rspf | derived | 0.92 | 42.8 | 47.8 | 0.60 | 14.0 | 0.4 |

Source: `data/reports/smoothing/metrics_summary.csv` (subgroups, IQRs and per-patient
values are there too), `methods.html` and the dashboard.

What the table says:
- **Derived settings move nearly every method toward the ideal**: higher correlation,
  smaller residual spread, better peak preservation. logmhw is the exception.
- **Defaults can be badly wrong.** gpssm at default removes 97.5% of the variance with
  a correlation of 0.17 — it flattens the signal. Derived, it is the closest to the
  ideal on correlation and residual spread.
- **There is a real trade-off**: removing more jitter costs peaks. ossa derived keeps
  almost every peak (0.91) but barely smooths (15% variance removed); krlst default
  smooths hard and keeps few peaks. The ideal sits in between.
- Closest to the ideal overall (derived): **rspf, ctimm, gpssm, hrkf** — but see causality.

## 6. Causality — the constraint that matters for early warning

A warning must be computed from data available before the event. `selftest` probes this
by truncation: does a run on a prefix reproduce the full run where they overlap?

| method | deviation when truncated (0 = causal) |
|---|---|
| anrewma | **0** — the only one causal end to end |
| ctimm | 0.10 |
| kim | 0.11 |
| gammadglm | 0.13 |
| krlst | 0.27 |
| gpssm | 0.33 |
| hrkf | 0.36 |
| ossa | 0.74 |
| logmhw | 1.73 |
| rspf | 3.16 |

None of the ten uses a backward pass any more. The nine non-zero ones filter forward
but fit their per-patient parameters (noise level, log-mean, observed range, median
step) on the whole recording, so a residual at time t has seen later data. That makes
their residuals optimistically biased for any onset claim. Rule for now: **use anrewma
for onset results**, or report the others as biased. The fix — fitting those parameters
on an expanding window — is planned, not done.

Note the tension: the methods closest to the ideal (rspf, gpssm, hrkf) are also among
the least causal.

## 7. HR

The same ten smoothers run on HR unchanged (`smooth <method> --signal hr`); each
channel is segmented on its own readings. The round-1 study and its derived settings
were done on HRV only. HR has 17× more readings per day but only ~3× more independent
information (correlation time ~9 min), heavy-tailed residuals (excess kurtosis ~9:
short surges from posture and movement), and a strong day/night swing in variability.
The derived settings have not been re-derived for HR — that is part of round 2.

## 8. What round 2 will do

1. **Judge smoothers by what they do for the model**, not only by signal metrics:
   `sweep smoothing` runs every smoother through the same model, folds and test patients.
2. **Per channel**: derive HR settings the way HRV's were, and compare.
3. **Make the leading candidates causal** (expanding-window parameter fits), then re-check
   the truncation test.
4. **Avoid tuning leakage**: settings derived from the whole cohort use the test
   patients' data; for final numbers, derive them inside the training folds.
5. **Re-adjust after the models**: once paths 4 and 5 report against the labels, revisit
   which smoother and settings serve detection and onset best — they may differ.

## 9. Where things are

| what | where |
|---|---|
| methods | `wearable/smoothing/<name>.py` |
| smooth step | `wearable/steps/smooth.py` (`python -m wearable smooth <method> [--signal hr]`) |
| cohort statistics | `wearable/cohort_stats.py` |
| the study | `scripts/smoothing_study.py`; results in `data/reports/smoothing/` |
| method write-ups | `docs/RSPF_method.md`, `docs/HRV_Smoothing_Methods_Explained.docx`, `docs/Smoothening.pdf` |
| causality probe | `python -m wearable selftest` (the `causality` line) |
