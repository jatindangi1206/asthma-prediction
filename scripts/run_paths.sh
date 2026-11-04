#!/usr/bin/env bash
# Runs the protocol's paths one after another, each for detection and next-day onset,
# over the 7 -> 1 day window ablation, 5 seeds, 80:20 split (config defaults).
# Results accumulate in data/reports/experiments/{runs,summary}_<task>.csv; one log
# per path and task in data/logs/paths/. Everything is cached, so a rerun after a
# crash resumes where it stopped. Paths can run as parallel streams (`run_paths.sh 1 4 5 &`,
# `run_paths.sh 6 &` ...); the CSVs are rewritten at the end of each sweep, so finish
# with one sequential `run_paths.sh` — every cell is a cache hit — to record them all.
#
#   scripts/run_paths.sh            # all paths
#   scripts/run_paths.sh 1 5        # just these
set -u
cd "$(dirname "$0")/.."
export PYTHONNOUSERSITE=1
PY=.venv/bin/python
LOG=data/logs/paths
mkdir -p "$LOG"
MODELS=always_no,xgboost,catboost,rnn,lstm

variants() {
  case $1 in
    1) echo raw ;;
    2) echo smoothed@anrewma,smoothed@ctimm,smoothed@gammadglm,smoothed@gpssm,smoothed@hrkf,smoothed@kim,smoothed@krlst,smoothed@logmhw,smoothed@ossa,smoothed@rspf ;;  # every smoother
    3) echo raw+cpd ;;
    4) echo smoothed+cpd ;;
    5) echo residual+cpd ;;
    6) echo surprise+cpd ;;          # expands to every forecaster
    7) echo embed ;;                 # expands to every encoder
  esac
}

run() {  # path task [extra args]
  local p=$1 task=$2; shift 2
  echo "$(date -Is) path $p $task start" | tee -a "$LOG/progress.log"
  $PY -m wearable "$@" > "$LOG/path${p}_${task}.log" 2>&1
  echo "$(date -Is) path $p $task exit $?" | tee -a "$LOG/progress.log"
}

for p in "${@:-1 2 3 4 5 6 7 8}"; do
  for p in $p; do
    if [ "$p" = 8 ]; then
      # MantisV2 as the classifier: the recorded series, 7 days at 55 slots a day
      for task in detection onset; do
        h=""; [ $task = onset ] && h="--horizon 1"
        run 8 $task sweep inputs --task $task $h --variants raw --days 7 \
            --profile 55 --channels hrv,hr,temp --model_names mantis_clf,xgboost,always_no --boot 200
      done
      continue
    fi
    v=$(variants "$p")
    run "$p" detection sweep window --task detection --variants "$v" --model_names $MODELS --boot 200
    run "$p" onset sweep window --task onset --horizon 1 --variants "$v" --model_names $MODELS --boot 200
  done
done
