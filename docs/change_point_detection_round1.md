# Change-point detection — round 1

*Status as of 3 Nov 2026. This is the first round. The detectors, their settings and the
way their thresholds are set will be compared and validated, and adjusted once the
prediction models report against the exacerbation labels. Design note with the full
measurements: https://claude.ai/artifact/CDgdpCNWBYfiZrqGdNiVh9 (private; ask for access).*

## 1. What change-point detection (CPD) is for

The smoother gives each channel a baseline and a residual (`docs/smoothing_round1.md`).
CPD asks, reading by reading: **has the patient's physiology just changed?** — a shift
in level, a burst of volatility, a slow drift. Its alarms become day-level features
(minutes flagged, number of alarms, their size) that the model can use.

Where it sits in the experiment paths:

| path | what the detectors read |
|---|---|
| 3 | the raw signal (`raw+cpd`) |
| 4 | the smoothed baseline (`smoothed+cpd`) |
| 5 | the residual — **the main hypothesis** (`residual+cpd`) |
| 6 | a foundation model's forecast surprise (`surprise+cpd@<model>`) |

Five detectors vote (an alarm from any of them counts):

| detector | looks for |
|---|---|
| BOCPD | a new segment starting — a shift in mean or spread |
| KCP (kernel) | the last 3 h distributed differently from the 3 h before |
| Kalman | spikes, a sustained one-directional drift, loss of randomness |
| HMM | a sustained high-volatility regime |
| Drift (new) | tonight's sleep level moving away from the patient's recent nights |

## 2. Why the original detectors had to change

They were written for HRV at a 10-minute cadence and assumed independent Gaussian noise
with one fixed scale. Measured on this cohort, the residuals are none of that:

| property | HRV | HR |
|---|---|---|
| readings per worn day | 47 | 786 |
| correlation between consecutive residuals | 0.23 (10 min) | **0.78** (1 min) |
| independent pieces of information per worn day | ~22 | ~65 (3×, not 17×) |
| residual readings beyond 3σ (Gaussian: 0.27%) | 5.4% | **11.4%** |
| night vs day spread | ~2× lower at night | ~4× lower at night |

On pure noise with no change, autocorrelation like HR's alone took kcpd from 0 to 21
false alarms per 1,000 readings and BOCPD from 0.2 to 25. The result on real HR was
**37.7 alarms per worn day, with 48% of all HR minutes flagged**. HRV gave 0.8 per day.

Other problems found:
- Every time setting was counted in readings, so "3 hours" on HRV was 18 minutes on HR.
- The HMM fitted itself on the whole segment and decoded with Viterbi, so a reading's
  flag depended on later data — and, across midnight, on the next day. That leaks into
  the onset task.
- BOCPD's documented trigger ("P(run length = 0) > 0.95") can never fire; with a
  constant hazard that probability always equals the hazard.

## 3. Decisions taken (2–3 Nov 2026)

| # | decision |
|---|---|
| 1 | HRV readings of 120–129 are ordinary data — nothing excluded or censored |
| 2 | a level jump across a gap is recorded as a column, never raised as an alarm |
| 3 | sensitivity is set by an alarm budget on no-change data; three levels are compared |
| 4 | a slow nightly-drift detector is added to the default set |
| 5 | everything is causal (real-time valid), for both tasks |
| 6 | no population prior: each patient's own first worn day starts their baseline |
| 7 | readings are compared within the same behavioural state (sleep / active / rest), on top of the time of day |
| 8 | drift compares sleep periods, not a fixed clock night |
| 9 | STS simulation is added as a second calibration method, beside day shuffling |

## 4. What was built

### 4.1 The noise model, in front of every detector (`wearable/cpd/_whiten.py`)

Each patient and channel gets a model of their own normal noise, built only from the
past, which turns the residual into **surprise scores (z)** that behave like plain
random noise when nothing is changing:

1. **HR is binned to 5-minute medians first.** Its correlation time is ~9 min, so little
   is lost, and medians absorb 1–5-minute surges from posture and movement.
2. **Arming.** Nothing alarms before the patient's first 24 worn hours.
3. **Centre and scale by time of day and state.** A smoothed 24-hour profile, corrected
   for state: sleep (`sleep_active`), active (awake with steps), rest (awake, no steps).
   We measured both: clock hour explains the residual's spread about twice as well as
   state, but state adds to it (sleep at night and daytime naps differ), so it is
   hour × state, not one or the other.
4. **Whitening.** A robust Kalman filter on an OU + white-noise model removes the
   short-term memory; a long gap decays it, a reset at a ≥ 180-min gap clears it.
5. **Refit every midnight** from the trailing 14 days, skipping abnormal days (more than
   30% of |z| > 3), so a real decline cannot become the new normal.
6. **Settle windows** after the watch comes back on (HR 30 min after a ≥ 15-min gap; HRV
   20 min after a ≥ 60-min gap): detectors update but cannot alarm. The level jump across
   the gap goes to `reentry_jump_z` / `reentry_gap_min`.

### 4.2 The detectors (same output columns and phenotypes as before)

| detector | what changed |
|---|---|
| BOCPD | models mean *and* spread (Student-t predictive); the chance of a change grows with time since the last reading; log space and pruning; scored as the probability that a new segment began in the last 60 min |
| KCP | 3-hour windows in clock time that never cross a reset; each half must be ≥ 70% full; every 30 min |
| Kalman | watches the noise model's innovations: spike, two-sided CUSUM drift, loss of whiteness over 2 h — three statistics, three thresholds |
| HMM | forward filter only (no Viterbi, no EM); dwell times in minutes (calm ~24 h, volatile ~2 h); Student-t emissions, so one surge does not flip it |
| Drift (new) | median level during sleep (18:00–12:00 window), CUSUM against the patient's last 28 nights; restarts after each alarm; flags once per verdict; falls back to 00:00–06:00 without sleep flags |

All settings are in minutes, separately for HRV and HR (`config.CPD_SETTINGS`;
`python -m wearable list` prints them).

## 5. Thresholds — how sensitive each detector is

### 5.1 The idea

We have no labels for "a change point happened", so thresholds cannot be tuned to hit
real changes. Instead, each detector's **jumpiness** is fixed: how often it fires on data
where we *know* nothing changed. Every patient and both channels then have equally jumpy
detectors, and an alarm means the same thing for everyone.

- The **budget** is how often the whole ensemble may fire on no-change data:
  **1 per 7, 1 per 3 (default), or 1 per worn day** — all three are built and compared
  (`--cpd_budget 0.142857,0.333333,1.0`).
- It is split evenly across the five detectors, then across a detector's statistics.
- A threshold counts **complete alarm events**, not readings over the line.
- Thresholds are rebuilt every 7 days from trailing history (before day 8: the first
  worn day) and only ever apply forward — causal.

### 5.2 Two ways to make the no-change data (compare, then choose)

| | shuffle (default) | STS (new) |
|---|---|---|
| idea | resample whole days of the patient's own past in shuffled order | fit a model of the patient's normal behaviour on the past and simulate new data |
| keeps | everything within a day exactly (bursts, tails, sleep, gaps); real day-to-day variation | the patient's timestamps, gaps and states exactly; their level, 24-h shape, autocorrelation, state shifts and tail weight |
| destroys | any trend across days | nothing real is reused — values are new draws |
| goes through | the same noise model and detectors | the same smoother, residual, binning, noise model and detectors |
| code | `_whiten.twin` | `wearable/cpd/_sts.py` |

**The STS model** (per patient and channel, refitted at each recalibration):
`log y = intercept + 24-h harmonics + AR(2) + b_sleep·sleep + b_active·active + noise`,
with Student-t noise whose tail is fitted to the patient's forecast errors. It runs on a
clock-time grid (HR 1 min, HRV 10 min) with the gaps left missing. Choices and why:

- **Stationary, not a random-walk level.** The textbook model (wandering level) was
  tried first and rejected on the data: simulated HR spread over 50–190 bpm in a week
  against a real 80–115. AR(2) keeps slow variation but returns to the patient's norm.
- **No outlier removal**: heavy tails go into the t noise instead.
- **Independent weekly blocks**, one empty day apart, on the history's own mask.
- **Fallbacks**: under 3 days of history, no 24-h term; if a fit fails, shuffle is used
  for that recalibration and recorded as such.

Fidelity check on two patients, 7-day history: percentiles and per-state medians match
closely (HR 80/96/115 real vs 79/96/114 synthetic; HRV asleep 30 vs 31). Weakness: HR's
60-minute autocorrelation is 0.43 synthetic vs 0.62 real.

Both methods write a `calibration` table beside the detections — per patient and
recalibration: method used, fitted STS parameters, every threshold, and the alarm rate
achieved on the no-change data. `scripts/cpd_calibration_compare.py` compares them.

## 6. Results so far

### 6.1 Full cohort, default budget (target: 0.33 alarms per worn day on no-change data)

| | patients | worn days | alarms per worn day | time flagged |
|---|---|---|---|---|
| HR — original detectors (5-patient check) | 5 | — | 37.7 | 48% |
| HR — new, shuffle | 190 | 7,062 | **1.40** | **9.8%** |
| HRV — original detectors (5-patient check) | 5 | — | 0.8 | 2.3% |
| HRV — new, shuffle | 177 | 4,275 | **1.36** | **8.5%** |
| HRV — new, STS | 177 | 4,275 | **1.40** | **7.5%** |
| HR — new, STS | 190 | 7,062 | **5.05** | **25.0%** |

By detector (alarms per worn day; target 0.067 each on no-change data):

| | BOCPD | KCP | Kalman | HMM | Drift |
|---|---|---|---|---|---|
| HRV shuffle | 0.33 | 0.38 | 0.69 | 0.49 | 0.18 |
| HRV STS | 0.37 | 0.26 | 0.65 | 0.73 | 0.18 |
| HR shuffle | 0.20 | 0.46 | 0.62 | 0.56 | 0.23 |
| HR STS | 2.07 | 1.40 | 2.74 | 2.98 | 0.23 |

Recalibrations by method: HRV STS 1,528 full / 69 short-history / **524 (25%) fell
back to shuffle** (too few readings in the window); HR STS 2,407 / 151 / 200 (7%).

**STS vs shuffle, round-1 verdict.** On HRV they are equivalent (1.40 vs 1.36 alarms
per day; STS shifts alarms from KCP to the HMM). On HR STS is clearly worse: its
thresholds are much lower (median Kalman spike 6.1 vs 14.5, KCP 8.6 vs 21.5), so real HR
fires 3.6× more often. Its synthetic HR is tamer than real HR: the fitted tail weight
sits at its floor (ν = 3) for every patient, and AR(2) misses HR's bursts and
hours-scale wander. Shuffle stays the default. STS would need a better HR model
(heavier or bursty noise, a second slow component) before it can compete there.

**A shuffle issue found by the comparison:** on HR, BOCPD's threshold sits at its
ceiling (34.5 on the log-odds scale) in most recalibrations, with no alarms at all on
the no-change data — BOCPD is nearly switched off for HR under shuffle. Round 2 should
find out why the shuffled-day twin saturates it.

Per-statistic thresholds, achieved no-change rates and the fitted STS parameters
(median and IQR over patients) are in `data/reports/cpd_calibration/`. Fitted STS
parameters, median over patients: HRV AR(1) 0.51 per 10 min, HR 0.92 per minute; the
sleep effect is small for most patients (median −0.03 on log-HRV, −0.009 on log-HR),
far smaller than in patients with good sleep coverage (a021: −0.35).

**Reading these numbers.** On no-change data each threshold hits its target; on real data
alarms run about 4× the budget. Part of that is real change — which is what the detectors
are for — and part is what the no-change data cannot reproduce. So the budget is a
*sensitivity setting* that orders the three builds, not a guaranteed alarm rate. Even
on data generated from the STS model itself, real rates came out ~3× the budget, so
this is a property of calibrating on limited history, not of one method.

### 6.2 Five patients, by eye (`data/reports/plots/cpd/`)

- **a021, onset 8 Jan:** HRV fell from ~80 to ~30 overnight; BOCPD, Kalman and HMM fired
  within minutes on HRV, HR's Kalman fired that morning and HR drift at noon.
- **c001, onset 19 Jun:** HR Kalman and KCP fired that evening; HRV had no data.
- **But alarms are spread across whole recordings**, not concentrated near
  exacerbations. Whether they carry signal is a question for the model and the labels.
- The sleep flags come in short blocks (often under an hour a night), and HRV is sparse
  during sleep — the state correction and drift lean on little data on many nights.

### 6.3 What is guaranteed (`python -m wearable selftest`)

- the whole detect step is **causal**: a run on a prefix of the data reproduces the full
  run exactly — for shuffle and for STS;
- the state correction equalises z's spread across sleep and wake (0.69 → 1.02);
- STS preserves gaps and states exactly, is reproducible, recovers a known model's
  autocorrelation (0.42 vs 0.43) and sleep effect (−0.44 vs −0.35);
- settings are typo-checked; the default calibration's cache keys are unchanged.

## 7. Known limitations

- Real alarm rates are ~4× the budget (above); the first week runs hotter, because
  thresholds there come from a single worn day (the cost of decision 6).
- STS over-fires on HR (5.05 vs 1.40 alarms per day) and falls back to shuffle in a
  quarter of HRV recalibrations; its AR(2) mostly settles on one timescale, so hours-scale autocorrelation is under-modelled; it cannot
  reproduce the HRV pile-up at 120–129; `--on surprise` is not supported (it would need
  the foundation model run on every simulation).
- Detection runs on a residual from a smoother that is not causal unless it is anrewma
  (see `docs/smoothing_round1.md` §6).
- The sleep/activity flags are coarse.

## 8. Round 2 — validate, compare, choose

1. **Injection test:** add known changes (mean jump, variance jump, slow drift, a change
   inside a gap) to real quiet stretches; measure detection probability and delay per
   detector, channel and calibration.
2. **Against the labels:** alarm rate in the days before onset vs ordinary days, with
   patient-level confidence intervals; then AUROC inside paths 3–5.
3. **Choose on training folds only:** budget (1/7, 1/3, 1), calibration (shuffle vs STS),
   which detectors per channel, and the smoother underneath.
4. **Adjust after the models:** settings that look right on signal behaviour may not be
   the ones the models need; revisit once paths 4 and 5 report.

## 9. Where things are

| what | where |
|---|---|
| noise model, twin, thresholds | `wearable/cpd/_whiten.py` |
| STS calibration | `wearable/cpd/_sts.py` |
| detectors | `wearable/cpd/{bocpd,kcpd,kalman,hmm,drift}.py` |
| detect step | `wearable/steps/detect.py` |
| settings | `wearable/config.py`: `CPD_SETTINGS`, `CPD_BUDGETS`, `CPD_CALIBRATION`, `STS_SETTINGS` |
| run | `python -m wearable detect --on residual --signal hr --calibration sts --budget 0.333` |
| sweep | `python -m wearable sweep inputs --variants residual+cpd --cpd_budget 0.142857,0.333333,1.0 --cpd_calibration shuffle,sts` |
| compare calibrations | `python scripts/cpd_calibration_compare.py` → `data/reports/cpd_calibration/` |
| plots | `data/reports/plots/cpd/` |
| tests | `python -m wearable selftest`; `tests/test_smoke.py` |
