#!/bin/bash
# V32 full pipeline (repository root): nohup caffeinate -i bash training/v32-strong/pipeline.sh &
set -e
L=training/v32-strong
python3 $L/run.py pool > $L/pool.log 2>&1
python3 $L/run.py freeze-pool
git add $L && git commit -q -m "Freeze V32 confirmation pool before training" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
python3 $L/run.py train T3 > $L/train-T3.out 2>&1 &
python3 $L/run.py train T5 > $L/train-T5.out 2>&1 &
wait %1 && wait %2
python3 $L/run.py freeze-models
git add $L && git commit -q -m "Freeze V32 trained models before any evaluation" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
python3 $L/run.py screen > $L/screen.log 2>&1
python3 $L/analyze.py screen > /dev/null
python3 $L/run.py confirm > $L/confirm.log 2>&1
python3 $L/analyze.py confirm > /dev/null
echo done > $L/pipeline.done
