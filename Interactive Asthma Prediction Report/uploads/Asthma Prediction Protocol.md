# Asthma Prediction Protocol

What the data is, what we ask of it, every route from signal to prediction, and where the results go. Written so that anyone — or any assistant — can pick the project up from this page alone.

## **0** Orientation

Repository: `asthma-prediction/`. **Run everything from that directory** — paths are relative to it. `python` below means `.venv/bin/python`; call it directly, because activating the venv does not always take in a non-interactive shell.

### The shape of the thing

Six pools of interchangeable methods, discovered from the filesystem. **Adding a method is one file in a pool directory** — there is no list to register in.

The six pools

| pool | what a method does | count |
| --- | --- | --- |
| smoothing/ | estimate the level at each reading → smoothed + residual | 10 |
| forecasting/ | predict the next readings → surprise + tail | 3 |
| cpd/ | flag change points in whichever signal it is given | 4 |
| encoders/ | turn a window into a vector per patient-day | 2 |
| models/ | predict the probability | 6 |
| metrics/ | score a prediction | 19 |

### Five rules the repo enforces

1. **Steps never import each other.** Only `experiment.py` connects them.
2. **Adding a method is one file.** Pools are read off the filesystem.
3. **Every result records how it was made** — method, settings, inputs, code version — in a `manifest.json` beside it.
4. **Nothing is computed twice.** Results are keyed on a content hash; a matching key is reused.
5. **A comparison varies only the method.** Same test patients, same model, same seed.

`python -m wearable selftest` asserts all five plus every invariant below, and touches no data. Run it first; it should print OK — all invariants hold.

### Where to look

Where to look

| file | what is in it |
| --- | --- |
| config.py | every constant, path, channel and split setting — the only definitions |
| windowing.py | the window rule, shared by `join` and `encode`. Never redefine it |
| chunking.py | the gap rule: a break of 180 minutes or more starts a new segment |
| steps/ | ingest · smooth · forecast · detect · encode · join · model · evaluate |
| experiment.py | the only thing that connects steps: the paths and the sweeps |
| selftest.py | what must stay true. Read this to learn what the repo guarantees |
| data/runs/ | every stored result, keyed by content, with its provenance |
| data/reports/ | regenerable output — tables and figures |

**The HRV data is sensitive.** Nothing under `data/` is ever committed — not the raw cohorts, not the stored results, not any patient file.

## **1** Data

| cohort | patients |
| --- | --- |
| AIIMS adults | 110 |
| AIIMS paediatric | 54 |
| COPD | 27 |
| total usable | 191 |

### Patients

|  | patients | days |
| --- | --- | --- |
| with exacerbation | 38 | 4,976 |
| without | 153 | 10,715 |
| total | 191 | 15,691 |

### Days

15,691

patient-days

144

exacerbation days · 0.92 %

15,547

normal days

57

scoreable episodes

The bar is to scale: one day in 109 is positive. **Episodes, not days, are the real sample size** — about 10 land in each test set.

| group | patients | days | exac days | exac patients |
| --- | --- | --- | --- | --- |
| aiims_adult | 110 | 9,338 | 125 | 26 |
| aiims_paed | 54 | 4,981 | 8 | 6 |
| copd | 27 | 1,372 | 11 | 6 |

8 exacerbation days across 54 paediatric patients is why the patient split is stratified by group — an unstratified draw can leave a fold with no positive test patient at all.

**An unused option worth 42 % more positives.** The study's `sample` column means *"the day has at least one HRV reading"*, so days with heart rate but no HRV are marked unscoreable — and so are the episodes they belong to. Those days carry a median of **710 heart-rate readings each**. Widening the rule to *has HR or HRV* gives **20,050 days, 204 exacerbation days, 73 episodes, 221 patients**. Built as `join --valid any`, opt-in and never the default, because switching changes every number in the paper. Onset does not widen — its validity also requires "not inside an episode" and "not in the patient's last 3 days", which would have to be rederived.

## **2** The two questions

Kept in separate tables, never mixed. Both are asked of one patient-day at a time.

### Detection

question

Is day D an exacerbation day?

input

X days up to and including day D — `--days X`

output

probability that day D is an exacerbation day

asked on

15,691 days — 144 positive (0.92 %)

day D−6 … D*X = 7 days in*

→

model

→

P(day D is an\
exacerbation day)

**The ablation:** hold the question fixed and shrink the input — 7 days in, then 6, 5, 4, 3, 2, down to day D alone.

### Onset

question

Will a new exacerbation start on day D+1?

input

X days up to and including day D — identical to detection

output

probability that a new episode starts within H days after D

asked on

15,196 days — 151 positive (0.99 %) at the labels' 1–3 day horizon

day D−6 … D*X = 7 days in*

→

model

→

P(episode starts\
on day D+1)*--horizon 1*

**Two axes.** X shrinks 7→1 as above, *and* the question tightens: within 3 days, within 2, or tomorrow only. Tightening keeps the same days and narrows only the label, so the denominator never moves and two horizons compare directly.

## **3** Variables

**Every model gets every variable, always.** The question is never "can HRV alone predict this" — it is "can these measurements predict this". A path decides what happens to HR and HRV; every other variable goes in raw alongside.

| variable | coverage | what the model gets per day |
| --- | --- | --- |
| HR | 100 % | average, spread, lowest, highest — plus smoothed and residual on the smoothing paths |
| HRV | 6.1 % | the same |
| temp | 8.0 % | average, spread, lowest, highest |
| steps | 2.8 % | the same |
| spo2 | 21 total | left out — 21 readings in the whole cohort |
| sleep_active | 100 % | minutes asleep, and how many separate bouts |
| steps_active | 100 % | minutes active, and how many bouts |
| deep_sleep light_sleep almost_awake | per bout | total minutes that day, and the bout count |

### Each channel is processed on its own

Smoothing, forecasting and encoding run on **one channel at a time**, same pipeline, separate output. HR is not a covariate for HRV. The channels differ 16-fold in cadence, and a gap of 180 minutes or more segments a series on *its own* readings, so a shared grid would hide one channel's holes behind the other's.

HR*12.1M readings*

→

the same 10 smoothers*--signal hr*

→

smoothed_hr\
residual_hr

HRV*736K readings*

→

the same 10 smoothers*--signal hrv*

→

smoothed_hrv\
residual_hrv

They meet only at `join`, where each becomes its own block of day features.

### Sleep stages

The export gives one row per sleep bout with the minutes already totalled — for example 45 min deep, 20 min light, 0 almost awake. **It never says when deep sleep started or ended.** So no timeline is assumed and bout lengths are never averaged: a day gets the totals and the bout count, nothing more.

### Readings per unbroken segment

| channel | obs (median) | per unbroken segment | per day | fills a 512-point context? |
| --- | --- | --- | --- | --- |
| hr | 52,718 | **885** | 744 | yes |
| temp | 3,867 | 68 | 62 | no |
| hrv | 2,828 | 49 | 55 | no |

A foundation model needs its context *contiguous*, and MOMENT and MantisV2 both want 512 points. A 49-reading HRV segment cannot fill that; an 885-reading HR segment can. This is why the foundation routes run on HR.

## **4** Smoothing and residuals

raw signal

→

smoother*10 to choose from*

→

smoothed*the baseline*

→

residual*raw − smoothed*

Each channel yields **three** signals, not one — raw, the estimated baseline, and the residual. All three always travel into the model, and so do temp, steps and sleep, unsmoothed. What a path chooses is not which signal the model sees; it is **which one the change-point detectors read**.

**Only one smoother is causal end to end.** Nine of the ten set their per-patient settings from the whole recording, so a residual at time t has seen data from after t. Deviation when the run is truncated: rspf 3.16, logmhw 1.73, ossa 0.74, hrkf 0.36, gpssm 0.33, krlst 0.27, gammadglm 0.13, kim 0.11, ctimm 0.10, anrewma 0. **Use anrewma for any onset claim** — a warning is by definition made before the event — or report the others as optimistically biased.

## **5** Two more variables, from frozen models

Both replace a hand-built quantity with something a pretrained model computes. Neither is trained on this cohort.

### Surprise — forecast error

A frozen forecaster is shown the history and asked what comes next. The surprise is how wrong it was.

history*\~3.5 days*

→

frozen forecaster*predicts the next hours*

→

prediction

vs

actual

model expects = 65\
actual = 89\
surprise = 89 − 65 = +24

A smoother would *absorb* this — it is built to track a drifting baseline, so after two or three high readings its baseline rises and the residual shrinks. A forecaster predicting from before the change does not. It also reports where 89 fell in its own predictive distribution (tail, 0 to 1), which is comparable across patients in a way "+24" is not: 24 means different things at a 30 and a 90 baseline.

**And it is the only clean residual in the pipeline.** Frozen weights, nothing fitted on the cohort, each block predicted from strictly earlier readings — so unlike nine of the ten smoothers there is no lookahead to argue about.

### Embedding — learned representation

history*7-day window*

→

frozen encoder*MOMENT or MantisV2*

→

a vector*512 numbers per day*

→

model

Instead of choosing and relying on mean, median, SD, min and max, the encoder learns a representation of the **entire temporal pattern**. The 512 numbers are reduced inside the training fold, never before it — reducing earlier would fit on test patients.

**The context budget.** `days × profile` *is* the input length. MOMENT is fixed at 512, MantisV2 wants a multiple of 32 and was pretrained at 512, Timestral takes up to 32,768. So against a fixed ceiling, changing `--days` does not change how much the model is handed — it changes the resolution those slots are spent at. 7 days into 512 is \~20 minutes a slot; 1 day is \~2.8 minutes. HRV's 7-day window is 385 points and fits natively; HR's is 5,208 and needs `--profile 73`. `windowing.bins_for(7, 512)` returns 73, so the arithmetic is never implicit.

## **6** Why a foundation model is not simply used as the predictor

The obvious move is to hand the raw series to a large pretrained model and let it predict. That is not the main approach here, for reasons worth stating because they will be asked.

### Pretraining transfers the wrong knowledge

Pretraining teaches a model **what time series look like** — trends, rhythms, bursts, regime changes. It teaches it nothing about **what an asthma exacerbation looks like**. That mapping has to be learned from the labels, and there are **144 positive days**.

So the job splits by where the information comes from: signal knowledge from the big frozen model, which needs no labels; disease knowledge from a small classifier, which must learn a 144-example mapping and does so better for being small.

### "Direct" is mostly the same thing in disguise

144 positives cannot fine-tune a 110M–385M parameter body. So "use MOMENT directly" necessarily means freeze the body and train a small head — which *is* the embedding route, with the head restricted to one linear layer. Running embeddings into `xgboost` is strictly more general.

### The nearest evidence says it loses

The reference COPD design ran the equivalent experiment. From its own results tables:

| model | onset AUROC | detection AUROC |
| --- | --- | --- |
| TimesFM 3, frozen + learned logistic layer | 0.552 \[0.44–0.66\] | 0.499 |
| GAST — purpose-built, trained end to end | 0.506 | 0.502 |
| GAST with its gap-awareness switched off | 0.579 | 0.553 |
| gradient-boosted trees | among the better rows | — |

The frozen foundation model landed at exactly chance on detection. The purpose-built end-to-end network did no better, and was beaten by its own ablation.

### So why MantisV2 is still being tried

| reason |  |
| --- | --- |
| 8.1M parameters | The only model on the shortlist in the same universe as 144 positives. MOMENT-large is 47× bigger. End-to-end training here is a real option rather than memorisation |
| a native classifier | fit(X, y) / predict_proba(X) — not a head bolted onto a forecaster |
| cross-channel structure | It takes several channels at once and can *learn a combination of them*. Frozen per-channel embeddings cannot: each channel is embedded in ignorance of the others. This is the one capability the other routes genuinely lack |
| near-zero cost | Apache-2.0, CPU-feasible, one file. The result is informative either way, and a clean negative answers the question readers will ask |

Not Timestral, TimesFM or Chronos-2 for this: they only forecast. No classification head, and nothing exposed to attach one to.

## **7** Paths — the experiments

One path is one row of the comparison: an input, the models, a probability. **Every path feeds the model all the variables** — the path only decides what happens to HR and HRV.

**PATH 1** Raw baselinebuilt · run

input

X days up to and including day D

HR raw

HRV raw

temp

steps

sleep

activity

↓

models

always_no

xgboost

catboost

rnn

lstm

↓

output

one probability per test day

X goes 7, 6, 5, 4, 3, 2, 1. Nothing else counts as an improvement until this number exists.

**PATH 4** Smoothed + change pointsbuilt · run

input

HR & HRV smoothed

change points on the smoothed values

HR raw

HRV raw

temp

steps

sleep

activity

↓

models

always_no

xgboost

catboost

rnn

lstm

↓

output

one probability per test day

The baseline moves slowly, so regime shifts read cleanly — but the bursts have been smoothed away.

**PATH 5** Residual + change points main hypothesisbuilt · run

input

HR & HRV residual

change points on the residuals

HR raw

HRV raw

HR & HRV smoothed

temp

steps

sleep

activity

↓

models

always_no

xgboost

catboost

rnn

lstm

↓

output

one probability per test day

Drift and the patient's own level subtracted, the bursts kept. Paths 4 and 5 hand over the same columns — only the signal the detectors read differs, which is the comparison. Use anrewma for any onset claim.

**PATH 6** Surprise + change points tsfmbuilt · not run

input

HR & HRV surprise + tail

change points on the surprise

HR raw

HRV raw

temp

steps

sleep

activity

↓

models

always_no

xgboost

catboost

rnn

lstm

↓

output

one probability per test day

Path 5 with the baseline from a frozen foundation model instead of a fitted smoother — so nothing has seen the future. Forecaster: Timestral, TimesFM or Chronos-2, one at a time.

**PATH 7** Embedding tsfmbuilt · not run

input

a 512-number embedding of the 7-day window

HR raw

HRV raw

temp

steps

sleep

activity

↓

models

always_no

xgboost

catboost

rnn

lstm

↓

output

one probability per test day

No averages, no detectors. A frozen encoder decides what matters about the week. Encoder: MOMENT or MantisV2, one at a time.

**PATH 8** MantisV2 as the model the open questionbuilt · not run

input

the recorded series of HR, HRV and temp together, nothing else done to them

sleep and activity stay in the day features — at 0.3 % density, forcing them onto a shared grid would be inventing data

↓

model — the foundation model IS the model

mantis_clf\
8.1M, trained end to end

xgboost\
the row to beat

always_no

↓

output

one probability per test day

`models/mantis_clf.py`. Wiring verified on synthetic data; the real run is pending. fine_tuning_type=full trains the whole body (the experiment), head freezes it (the conservative reading) — both keyed, so they compare. Section 6 explains why this is the only direct-prediction route being tried.

### Every architecture in the pipeline

Eleven architectures, in two groups. **A classifier emits the probability**; a **feature producer** is frozen and emits a signal or a vector that a classifier then reads. Only `mantis_clf` is both.

| architecture | what it is | size | role | used by |
| --- | --- | --- | --- | --- |
| always_no | never flags a day | — | classifier | every path |
| xgboost | gradient-boosted trees, 400 × depth 4, blanks sent down a learned branch | — | classifier | every path |
| catboost | oblivious trees with ordered boosting, 400 × depth 4 | — | classifier | every path |
| rnn | forward RNN over the day window, hidden 64, 2 layers, read at the classified day | — | classifier | every path |
| lstm | the same with LSTM cells — gates that hold a slow drift across the window | — | classifier | every path |
| mantis_clf | MantisV2 patch transformer, trained end to end on the recorded series | 8.1M | classifier & foundation model | path 8 |
| timestral | Mistral Timestral — 24-layer patch transformer, width 768, patch 32, context to 32,768. Frozen | 209M | forecaster → surprise | path 6 |
| timesfm | Google TimesFM 2.5 — patch transformer. Frozen. 3.0 weights are non-commercial | \~200M | forecaster → surprise | path 6 |
| chronos | Amazon Chronos-2 — patch transformer with covariate support. Frozen | 120M | forecaster → surprise | path 6 |
| moment | MOMENT — T5-family encoder, patch 8, context fixed at 512, takes a missingness mask. Frozen | 110M base 385M large | encoder → embedding | path 7 |
| mantis | MantisV2 as a frozen encoder — the same body as `mantis_clf`, not trained | 8.1M | encoder → embedding | path 7 |

Paths 6 and 7 each run **one** foundation model at a time, so the comparison varies the model and nothing else. Paths 2 (smoothing alone) and 3 (raw + change points) exist in the code as --variants smoothed and raw+cpd; they are left off this page for now.

## **8** Phases — what one run computes

A phase is not an experiment. It is a stage of processing that runs **once**, over the whole cohort, and stores what it produced. The paths then pick from the shelf — nothing is recomputed per path, per model or per fold.

|  | phase | what it does | produces | built |
| --- | --- | --- | --- | --- |
| 1 | ingest | Read the export, lay each channel on its own time grid, write the labels. | all 10 channels, one file each | yes |
| 2 | smooth | Run the **10 smoothers** on HR, then on HRV — separately, same pipeline. | smoothed + residual, per channel | yes |
| 3 | forecast | Run the **3 frozen forecasters** on HR and HRV. | surprise · tail | built, not run |
| 4 | encode | Run the **2 frozen encoders** over each 7-day window. | an embedding per patient-day | built, not run |
| 5 | detect | Run the **4 detectors** on each signal the earlier phases produced — raw, smoothed, residual, surprise. | change-point features per signal | yes |
| 6 | join → model → score | Assemble a path's ingredients into one table per patient-day, split the patients 80:20, train, score. | one row per path, model and fold | yes |

ingest

→

smooth*10 × 2 channels*

→

forecast*3 × 2*

→

encode*2*

→

detect*4 per signal*

→

join · model · score*once per path*

A smoothed series, a surprise or an embedding does not depend on the path, the model, the fold or the seed — so it is computed once and reused by all of them. That is what makes a 0.3B encoder affordable, and why adding a path costs only phase 6.

### What never changes, in any phase or path

Fixed in every phase and path

| fixed |  |
| --- | --- |
| the patients | All 191. A patient a method could not handle gets blank numbers, not a removed day. |
| the split | **80:20**, whole patients to train or test and never both — the same patients every time. 5 seeds. |
| the variables | All of them reach the model, in every path. |
| the question | Detection and onset stay in separate tables. |
| the scoring | The same metrics, counted per day, per episode, and per probability band. |

The run asserts this: if two rows in a comparison were ever scored on different days, different positives or different patients, it stops rather than report them.

### Three analysis populations

The three analysis populations

| tier | --cohort | pts | days | exac days | base rate | role |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | all | 191 | 15,691 | 144 | 0.92 % | primary |
| 2 | smoothable | 187 | 15,685 | 144 | 0.92 % | robustness |
| 3 | exacerbators | 38 | 4,976 | 144 | 2.89 % | secondary, labelled |

Tier 2 excludes 4 patients carrying **6 days and 0 positives**; state it arithmetically or run it with `--stratify False`, because restricting the universe moves a patient across the split boundary in 3 of 5 seeds and a patient carries \~71 days. Tier 3 selects on the outcome, inflating the base rate 3.1× and making the false-alarm rate unmeasurable in the population that would be deployed to — a different question, never the headline.

## **9** How to run it

### Setup

\# python 3.10; call the venv binary directly\
python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt\
.venv/bin/pip install xgboost catboost # required: selftest fails on any method it cannot import\
.venv/bin/pip install mantis-tsfm # path 8 and the mantis encoder\
\
\# optional, opt-in foundation packages\
.venv/bin/pip install momentfm\
.venv/bin/pip install git+https://github.com/mistralai/timestral.git\
.venv/bin/pip install chronos-forecasting "timesfm\[torch\]"

Needs roughly **4 GB free** for the stored results and \~2 GB for the packages.

### Check, then compute

python -m wearable selftest # asserts every invariant; touches no data\
python -m wearable list # every discovered method and its settings\
\
python -m wearable ingest # phase 1 — all channels\
python -m wearable smooth gammadglm --signal hr # phase 2, one method\
python -m wearable forecast timestral --signal hr # phase 3\
python -m wearable encode moment --signal hr --days 7 --profile 73 # phase 4\
python -m wearable detect --on residual --smoothing gammadglm --signal hr # phase 5

### Then the experiments

\# paths 1, 4, 5 — detection then onset\
python -m wearable sweep inputs --task detection \\\
--variants raw,smoothed+cpd,residual+cpd \\\
--model_names always_no,xgboost,catboost,rnn,lstm\
\
python -m wearable sweep inputs --task onset --horizon 1 \\\
--variants raw,residual+cpd --model_names always_no,xgboost,catboost\
\
\# the input ablation: 7 days down to 1\
python -m wearable sweep window --task detection\
\
\# paths 6 and 7\
python -m wearable sweep inputs --variants surprise+cpd,embed\
\
\# path 8 — MantisV2 as the model. Needs the recorded series, so --profile\
python -m wearable sweep inputs --variants raw --days 7 --profile 55 \\\
--channels hrv,hr,temp --model_names mantis_clf,xgboost,always_no\
\
\# iterate fast: --boot 0 skips the intervals and is 145x quicker\
python -m wearable sweep inputs --variants raw --boot 0

Tables land in `data/reports/experiments/` — runs\_\<task>.csv one row per cell, summary\_\<task>.csv averaged over seeds with the intervals.

### Measured cost

| stage | 191 patients | disk |
| --- | --- | --- |
| ingest, all channels | \~1 min | \~1.1 GB |
| smooth, all 10 methods | 2–3 min | \~1.3 GB |
| detect, 4 detectors | 5–12 min | 400 MB each |
| join, one fold | 3.3 s | 7 MB |
| model, one cell | 0.1–1 s | — |
| evaluate, 1000 draws | 2.9 s | — |
| evaluate, --boot 0 | 0.02 s | — |
| mantis_clf, one fold | \~1 h CPU, far less on GPU | — |

The bootstrap dominates — roughly 85 % of a full grid's wall clock. Iterate with `--boot 0` and turn it on once for the final tables.

## **10** Metrics

Seven numbers judge a prediction, and six more judge a smoother on its own. Nothing else is reported.

The four counts everything is built from

A day is **flagged** when its predicted probability is 0.5 or above. Every threshold metric below is one question about these four numbers.

|  | was an exacerbation day | was normal |
| --- | --- | --- |
| flagged | TP caught | FP false alarm |
| not flagged | FN missed | TN left alone |

Definitions

| TP | exacerbation day correctly flagged |
| --- | --- |
| FP | normal day incorrectly flagged |
| TN | normal day correctly identified |
| FN | exacerbation day missed |
| N | total evaluated days |
| p | predicted probability of exacerbation |
| y | actual outcome: 1 = exacerbation, 0 = normal |

### 10.1 Primary model performance

| metric | what it means in our experiment | formula | better | interval |
| --- | --- | --- | --- | --- |
| AUROC | How well does the model rank exacerbation days above normal days? | area under the ROC curve | higher | yes |
| Sensitivity | Out of all exacerbation days, how many did the model correctly identify? | TP / (TP + FN) | higher | yes |
| Specificity | Out of all normal days, how many did the model correctly identify? | TN / (TN + FP) | higher | yes |
| PPV | Out of all flagged days, how many were actually exacerbation days? | TP / (TP + FP) | higher | yes |
| Brier score | How close are the predicted probabilities to the actual outcomes? | (1/N) ∑ (p − y)2 | lower | yes |

AUROC is threshold-free — it only cares about the ordering, so it does not move if the 0.5 cut moves. Brier is the only one that judges the *probability* rather than the ranking. The other three all change with the threshold.

### 10.2 Clinical usefulness

| metric | what it means in our experiment | formula | better | read on |
| --- | --- | --- | --- | --- |
| Lead time | How many days before an exacerbation begins does the model first warn us? | onset date − first warning date | higher | onset |
| Detection delay | How many days after an exacerbation begins does the model first identify it? | first detection date − onset date | lower | detection |

Both read the onset date out of the episode id, which is \<patient>\_\<onset date> — for example a001_20251203. They are exact mirrors, so lead + delay = 0 on any set of predictions, and the selftest asserts it. Reported as the median across caught episodes; undefined when nothing was caught.

### 10.3 Signal metrics — for the smoothers only

These judge a smoother directly against the raw trace, with no model and no labels. Averaged per patient and then across patients, so one long recording cannot dominate — and they are the only scores available for a patient with no exacerbation record.

| metric | what it means | better |
| --- | --- | --- |
| correlation | how closely the smoothed trace follows the raw one | higher |
| variance_reduction | share of the variance the smoother removed | higher |
| roughness_reduction | share of the minute-to-minute jitter removed | higher |
| peak_preservation | 0–1: did it keep the bursts while removing the jitter? The tension with the two above is the point | higher |
| residual_std | spread of raw − smoothed | lower |
| mean_error_pct | average departure from the raw trace | lower |

### 10.4 The counts

The split is a pure function of (patients, seed, test share, strata), so these numbers are **deterministic** — they will come out the same on any machine. The positive count is the ceiling on TP: a model cannot catch more exacerbation days than the test set holds.

The cohort — fixed

|  | detection | onset |
| --- | --- | --- |
| patients | 191 | 188 |
| patient-days | 15,691 | 15,196 |
| positive days | 144 | 151 |
| normal days | 15,547 | 15,045 |
| base rate | 0.92 % | 0.99 % |
| episodes | 57 | 56 |

38 of the 191 patients ever have an exacerbation. 80:20 puts 38 patients in test and 153 in train.

Detection — the split, per seed

| seed | train days | train pos | train eps | test days | test pos = max TP | test eps |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 12,889 | 124 | 47 | 2,802 | 20 | 10 |
| 1 | 12,593 | 125 | 47 | 3,098 | 19 | 10 |
| 2 | 12,669 | 109 | 43 | 3,022 | 35 | 14 |
| 3 | 12,875 | 115 | 48 | 2,816 | 29 | 9 |
| 4 | 12,984 | 114 | 44 | 2,707 | 30 | 13 |

Onset — the split, per seed

| seed | train days | train pos | train eps | test days | test pos = max TP | test eps |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 12,489 | 128 | 48 | 2,707 | 23 | 8 |
| 1 | 12,174 | 126 | 47 | 3,022 | 25 | 9 |
| 2 | 12,281 | 110 | 42 | 2,915 | 41 | 14 |
| 3 | 12,482 | 127 | 47 | 2,714 | 24 | 9 |
| 4 | 12,590 | 117 | 43 | 2,606 | 34 | 13 |

**The test positives nearly double between seeds** — 19 on seed 1 against 35 on seed 2 for detection, 23 against 41 for onset. Whole patients move, and an exacerbator carries all of their days with them. So part of the spread between seeds is the test set being easier or harder, not the model changing. This is why a result is read as a mean across the five with its interval, never from one seed, and why the run asserts that every model in a fold saw *identical* test days, positives, episodes and patients.

### 10.5 What gets compared — fill in from the run

One row per **path × architecture**, per task, per seed. The raw rows land in `data/reports/experiments/runs_<task>.csv` and the averaged ones in `summary_<task>.csv`. `always_no` is the row that fixes what every other number means.

| path | architecture | TP | FP | TN | FN | AUROC | sens | spec | PPV | Brier | delay / lead |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 raw | always_no | — | — | — | — | — | — | — | — | — | — |
| 1 raw | xgboost | — | — | — | — | — | — | — | — | — | — |
| 1 raw | catboost · rnn · lstm | — | — | — | — | — | — | — | — | — | — |
| 4 smoothed+cpd | all 5 classifiers | — | — | — | — | — | — | — | — | — | — |
| 5 residual+cpd | all 5 classifiers | — | — | — | — | — | — | — | — | — | — |
| 6 surprise+cpd | × timestral | — | — | — | — | — | — | — | — | — | — |
| 6 surprise+cpd | × timesfm | — | — | — | — | — | — | — | — | — | — |
| 6 surprise+cpd | × chronos | — | — | — | — | — | — | — | — | — | — |
| 7 embedding | × moment | — | — | — | — | — | — | — | — | — | — |
| 7 embedding | × mantis | — | — | — | — | — | — | — | — | — | — |
| 8 tsfm as model | mantis_clf | — | — | — | — | — | — | — | — | — | — |

On paths 6 and 7 the foundation model is a second axis: each one produces its own features, and all five classifiers then read them. So those rows multiply out — three forecasters × five classifiers on path 6, two encoders × five on path 7.

## **11** Results

One block per path; inside it the two tasks; inside each task the **seven metrics of section 10**, each as an ablation curve. All curves share the same x-axis — days of input, 1 to 7 — so any metric reads across paths.

**No interpretation is offered.** The method is in sections 1 to 9, the metrics in section 10, the denominators in every caption. The reader judges.

How to add a figure

Save it to `data/reports/figures/` under the filename the slot names, then replace the dashed placeholder with one line:

\<img src="figures/path1_detection_auroc.png"\
alt="Path 1 detection AUROC vs days of input">

Keep the `<figcaption>`. Figures are published alongside the page, so the `src` stays a plain relative path.

Six slots per task

Five primary metrics on every task, plus the one clinical metric that applies: **detection delay** on detection, **lead time** on onset. 72 slots across the six paths.

The five primary ones carry a 95 % interval from resampling *test patients* — not days, because days within a patient are dependent and resampling them gives an interval several times too narrow. Lead time and detection delay are medians over caught episodes and are reported as point estimates.

**Read against this.** Six paths × two tasks × seven metrics × seven window lengths is several hundred numbers, against about **10 test episodes per fold**. Some will separate by chance. Pre-register one primary comparison before running, and report how many were made.

### 11.1 Path 1 — Raw baseline built · run

detection 15,691 days · 144 positive · base rate 0.92 %

Primary performance

**AUROC**\
path1_detection_auroc.png\
x: 1–7 days in

**AUROC** vs days of input · path 1 · detection. Shaded band = 95 % interval.

**Sensitivity**\
path1_detection_sensitivity.png\
x: 1–7 days in

**Sensitivity** vs days of input · path 1 · detection. Shaded band = 95 % interval.

**Specificity**\
path1_detection_specificity.png\
x: 1–7 days in

**Specificity** vs days of input · path 1 · detection. Shaded band = 95 % interval.

**PPV**\
path1_detection_ppv.png\
x: 1–7 days in

**PPV** vs days of input · path 1 · detection. Shaded band = 95 % interval.

**Brier**\
path1_detection_brier.png\
x: 1–7 days in

**Brier** vs days of input · path 1 · detection. Shaded band = 95 % interval.

Clinical usefulness

**Detection delay**\
path1_detection_detection_delay.png\
x: 1–7 days in

**Detection delay** vs days of input · path 1 · detection. Median over caught episodes; point estimate.

onset 15,196 valid days · 151 positive · --horizon 1

Primary performance

**AUROC**\
path1_onset_auroc.png\
x: 1–7 days in

**AUROC** vs days of input · path 1 · onset. Shaded band = 95 % interval.

**Sensitivity**\
path1_onset_sensitivity.png\
x: 1–7 days in

**Sensitivity** vs days of input · path 1 · onset. Shaded band = 95 % interval.

**Specificity**\
path1_onset_specificity.png\
x: 1–7 days in

**Specificity** vs days of input · path 1 · onset. Shaded band = 95 % interval.

**PPV**\
path1_onset_ppv.png\
x: 1–7 days in

**PPV** vs days of input · path 1 · onset. Shaded band = 95 % interval.

**Brier**\
path1_onset_brier.png\
x: 1–7 days in

**Brier** vs days of input · path 1 · onset. Shaded band = 95 % interval.

Clinical usefulness

**Lead time**\
path1_onset_lead_days.png\
x: 1–7 days in

**Lead time** vs days of input · path 1 · onset. Median over caught episodes; point estimate.

### 11.4 Path 4 — Smoothed + change points built · run

detection 15,691 days · 144 positive · base rate 0.92 %

Primary performance

**AUROC**\
path4_detection_auroc.png\
x: 1–7 days in

**AUROC** vs days of input · path 4 · detection. Shaded band = 95 % interval.

**Sensitivity**\
path4_detection_sensitivity.png\
x: 1–7 days in

**Sensitivity** vs days of input · path 4 · detection. Shaded band = 95 % interval.

**Specificity**\
path4_detection_specificity.png\
x: 1–7 days in

**Specificity** vs days of input · path 4 · detection. Shaded band = 95 % interval.

**PPV**\
path4_detection_ppv.png\
x: 1–7 days in

**PPV** vs days of input · path 4 · detection. Shaded band = 95 % interval.

**Brier**\
path4_detection_brier.png\
x: 1–7 days in

**Brier** vs days of input · path 4 · detection. Shaded band = 95 % interval.

Clinical usefulness

**Detection delay**\
path4_detection_detection_delay.png\
x: 1–7 days in

**Detection delay** vs days of input · path 4 · detection. Median over caught episodes; point estimate.

onset 15,196 valid days · 151 positive · --horizon 1

Primary performance

**AUROC**\
path4_onset_auroc.png\
x: 1–7 days in

**AUROC** vs days of input · path 4 · onset. Shaded band = 95 % interval.

**Sensitivity**\
path4_onset_sensitivity.png\
x: 1–7 days in

**Sensitivity** vs days of input · path 4 · onset. Shaded band = 95 % interval.

**Specificity**\
path4_onset_specificity.png\
x: 1–7 days in

**Specificity** vs days of input · path 4 · onset. Shaded band = 95 % interval.

**PPV**\
path4_onset_ppv.png\
x: 1–7 days in

**PPV** vs days of input · path 4 · onset. Shaded band = 95 % interval.

**Brier**\
path4_onset_brier.png\
x: 1–7 days in

**Brier** vs days of input · path 4 · onset. Shaded band = 95 % interval.

Clinical usefulness

**Lead time**\
path4_onset_lead_days.png\
x: 1–7 days in

**Lead time** vs days of input · path 4 · onset. Median over caught episodes; point estimate.

### 11.5 Path 5 — Residual + change points main hypothesis built · run

detection 15,691 days · 144 positive · base rate 0.92 %

Primary performance

**AUROC**\
path5_detection_auroc.png\
x: 1–7 days in

**AUROC** vs days of input · path 5 · detection. Shaded band = 95 % interval.

**Sensitivity**\
path5_detection_sensitivity.png\
x: 1–7 days in

**Sensitivity** vs days of input · path 5 · detection. Shaded band = 95 % interval.

**Specificity**\
path5_detection_specificity.png\
x: 1–7 days in

**Specificity** vs days of input · path 5 · detection. Shaded band = 95 % interval.

**PPV**\
path5_detection_ppv.png\
x: 1–7 days in

**PPV** vs days of input · path 5 · detection. Shaded band = 95 % interval.

**Brier**\
path5_detection_brier.png\
x: 1–7 days in

**Brier** vs days of input · path 5 · detection. Shaded band = 95 % interval.

Clinical usefulness

**Detection delay**\
path5_detection_detection_delay.png\
x: 1–7 days in

**Detection delay** vs days of input · path 5 · detection. Median over caught episodes; point estimate.

onset 15,196 valid days · 151 positive · --horizon 1

Primary performance

**AUROC**\
path5_onset_auroc.png\
x: 1–7 days in

**AUROC** vs days of input · path 5 · onset. Shaded band = 95 % interval.

**Sensitivity**\
path5_onset_sensitivity.png\
x: 1–7 days in

**Sensitivity** vs days of input · path 5 · onset. Shaded band = 95 % interval.

**Specificity**\
path5_onset_specificity.png\
x: 1–7 days in

**Specificity** vs days of input · path 5 · onset. Shaded band = 95 % interval.

**PPV**\
path5_onset_ppv.png\
x: 1–7 days in

**PPV** vs days of input · path 5 · onset. Shaded band = 95 % interval.

**Brier**\
path5_onset_brier.png\
x: 1–7 days in

**Brier** vs days of input · path 5 · onset. Shaded band = 95 % interval.

Clinical usefulness

**Lead time**\
path5_onset_lead_days.png\
x: 1–7 days in

**Lead time** vs days of input · path 5 · onset. Median over caught episodes; point estimate.

### 11.6 Path 6 — Surprise + change points tsfm built · not run

detection 15,691 days · 144 positive · base rate 0.92 %

Primary performance

**AUROC**\
path6_detection_auroc.png\
x: 1–7 days in

**AUROC** vs days of input · path 6 · detection. Shaded band = 95 % interval.

**Sensitivity**\
path6_detection_sensitivity.png\
x: 1–7 days in

**Sensitivity** vs days of input · path 6 · detection. Shaded band = 95 % interval.

**Specificity**\
path6_detection_specificity.png\
x: 1–7 days in

**Specificity** vs days of input · path 6 · detection. Shaded band = 95 % interval.

**PPV**\
path6_detection_ppv.png\
x: 1–7 days in

**PPV** vs days of input · path 6 · detection. Shaded band = 95 % interval.

**Brier**\
path6_detection_brier.png\
x: 1–7 days in

**Brier** vs days of input · path 6 · detection. Shaded band = 95 % interval.

Clinical usefulness

**Detection delay**\
path6_detection_detection_delay.png\
x: 1–7 days in

**Detection delay** vs days of input · path 6 · detection. Median over caught episodes; point estimate.

onset 15,196 valid days · 151 positive · --horizon 1

Primary performance

**AUROC**\
path6_onset_auroc.png\
x: 1–7 days in

**AUROC** vs days of input · path 6 · onset. Shaded band = 95 % interval.

**Sensitivity**\
path6_onset_sensitivity.png\
x: 1–7 days in

**Sensitivity** vs days of input · path 6 · onset. Shaded band = 95 % interval.

**Specificity**\
path6_onset_specificity.png\
x: 1–7 days in

**Specificity** vs days of input · path 6 · onset. Shaded band = 95 % interval.

**PPV**\
path6_onset_ppv.png\
x: 1–7 days in

**PPV** vs days of input · path 6 · onset. Shaded band = 95 % interval.

**Brier**\
path6_onset_brier.png\
x: 1–7 days in

**Brier** vs days of input · path 6 · onset. Shaded band = 95 % interval.

Clinical usefulness

**Lead time**\
path6_onset_lead_days.png\
x: 1–7 days in

**Lead time** vs days of input · path 6 · onset. Median over caught episodes; point estimate.

### 11.7 Path 7 — Embedding tsfm built · not run

detection 15,691 days · 144 positive · base rate 0.92 %

Primary performance

**AUROC**\
path7_detection_auroc.png\
x: 1–7 days in

**AUROC** vs days of input · path 7 · detection. Shaded band = 95 % interval.

**Sensitivity**\
path7_detection_sensitivity.png\
x: 1–7 days in

**Sensitivity** vs days of input · path 7 · detection. Shaded band = 95 % interval.

**Specificity**\
path7_detection_specificity.png\
x: 1–7 days in

**Specificity** vs days of input · path 7 · detection. Shaded band = 95 % interval.

**PPV**\
path7_detection_ppv.png\
x: 1–7 days in

**PPV** vs days of input · path 7 · detection. Shaded band = 95 % interval.

**Brier**\
path7_detection_brier.png\
x: 1–7 days in

**Brier** vs days of input · path 7 · detection. Shaded band = 95 % interval.

Clinical usefulness

**Detection delay**\
path7_detection_detection_delay.png\
x: 1–7 days in

**Detection delay** vs days of input · path 7 · detection. Median over caught episodes; point estimate.

onset 15,196 valid days · 151 positive · --horizon 1

Primary performance

**AUROC**\
path7_onset_auroc.png\
x: 1–7 days in

**AUROC** vs days of input · path 7 · onset. Shaded band = 95 % interval.

**Sensitivity**\
path7_onset_sensitivity.png\
x: 1–7 days in

**Sensitivity** vs days of input · path 7 · onset. Shaded band = 95 % interval.

**Specificity**\
path7_onset_specificity.png\
x: 1–7 days in

**Specificity** vs days of input · path 7 · onset. Shaded band = 95 % interval.

**PPV**\
path7_onset_ppv.png\
x: 1–7 days in

**PPV** vs days of input · path 7 · onset. Shaded band = 95 % interval.

**Brier**\
path7_onset_brier.png\
x: 1–7 days in

**Brier** vs days of input · path 7 · onset. Shaded band = 95 % interval.

Clinical usefulness

**Lead time**\
path7_onset_lead_days.png\
x: 1–7 days in

**Lead time** vs days of input · path 7 · onset. Median over caught episodes; point estimate.

### 11.8 Path 8 — MantisV2 as the model the open question built · not run

detection 15,691 days · 144 positive · base rate 0.92 %

Primary performance

**AUROC**\
path8_detection_auroc.png\
x: 1–7 days in

**AUROC** vs days of input · path 8 · detection. Shaded band = 95 % interval.

**Sensitivity**\
path8_detection_sensitivity.png\
x: 1–7 days in

**Sensitivity** vs days of input · path 8 · detection. Shaded band = 95 % interval.

**Specificity**\
path8_detection_specificity.png\
x: 1–7 days in

**Specificity** vs days of input · path 8 · detection. Shaded band = 95 % interval.

**PPV**\
path8_detection_ppv.png\
x: 1–7 days in

**PPV** vs days of input · path 8 · detection. Shaded band = 95 % interval.

**Brier**\
path8_detection_brier.png\
x: 1–7 days in

**Brier** vs days of input · path 8 · detection. Shaded band = 95 % interval.

Clinical usefulness

**Detection delay**\
path8_detection_detection_delay.png\
x: 1–7 days in

**Detection delay** vs days of input · path 8 · detection. Median over caught episodes; point estimate.

onset 15,196 valid days · 151 positive · --horizon 1

Primary performance

**AUROC**\
path8_onset_auroc.png\
x: 1–7 days in

**AUROC** vs days of input · path 8 · onset. Shaded band = 95 % interval.

**Sensitivity**\
path8_onset_sensitivity.png\
x: 1–7 days in

**Sensitivity** vs days of input · path 8 · onset. Shaded band = 95 % interval.

**Specificity**\
path8_onset_specificity.png\
x: 1–7 days in

**Specificity** vs days of input · path 8 · onset. Shaded band = 95 % interval.

**PPV**\
path8_onset_ppv.png\
x: 1–7 days in

**PPV** vs days of input · path 8 · onset. Shaded band = 95 % interval.

**Brier**\
path8_onset_brier.png\
x: 1–7 days in

**Brier** vs days of input · path 8 · onset. Shaded band = 95 % interval.

Clinical usefulness

**Lead time**\
path8_onset_lead_days.png\
x: 1–7 days in

**Lead time** vs days of input · path 8 · onset. Median over caught episodes; point estimate.

### 11.9 Across paths, and the smoothers on their own

Once a window length is settled from the curves above, these compare the paths at that one setting. The signal metrics judge a smoother against the raw trace, with no model and no labels involved.

**All paths · detection**\
paths_detection.png

**Detection, all paths** at the chosen window. One row per path × model; identical test patients in every row, asserted at run time.

**All paths · onset**\
paths_onset.png

**Onset, all paths** at --horizon 1.

**Calibration**\
calibration.png

**Calibration** — predicted against observed rate across 10 probability bands. Brier compresses this to one number; the curve shows where it goes wrong.

**By cohort group**\
by_group.png

**By group.** 125 exacerbation days among the adults against 8 among the children, so the paediatric interval will be wide.

**Analysis populations**\
cohorts.png

**Tier 1 vs 2 vs 3.** Tier 3's base rate is 3.1× tier 1's, so its numbers are not comparable in absolute terms.

**Smoothers, with a model**\
smoothers_model.png

**Ten smoothers** through the same model and folds. anrewma is the only one causal end to end.

The six signal metrics — a smoother judged with no model at all

**Correlation**\
signal_correlation.png

raw against smoothed. Higher better.

**Variance reduction %**\
signal_variance_reduction.png

variance removed. Higher better.

**Roughness reduction %**\
signal_roughness_reduction.png

jitter removed. Higher better.

**Peak preservation**\
signal_peak_preservation.png

0–1. Bursts kept while jitter removed.

**Residual std**\
signal_residual_std.png

spread of raw − smoothed. Lower better.

**Mean error %**\
signal_mean_error_pct.png

departure from the raw trace. Lower better.

Counts from `data/processed/labels_pass1.csv`. Channel coverage over 40 sampled patients (2,167,528 raw minute rows); segment lengths over 20; sleep columns over 12; timings measured with 7 workers. Feature counts from real `join` runs. Reference AUROCs are from the COPD design's own results tables. Build status as of 30 Oct 2026: paths 1, 4, 5 run; 6, 7, 8 built and not run. Seven metrics per prediction (5 primary + 1 clinical per task) and 6 signal metrics; `mantis_clf` verified on synthetic data only.