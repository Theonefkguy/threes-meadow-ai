#!/bin/bash
# V31 full pipeline (repository root): nohup caffeinate -i bash training/v31-stages/pipeline.sh &
set -e
L=training/v31-stages
python3 $L/run.py pools > $L/pools.log 2>&1
python3 $L/run.py sample > $L/sample.log 2>&1
python3 $L/run.py freeze-pools > $L/freeze-pools.out 2>&1
git add $L && git commit -q -m "Freeze V31 stage-1 pools before training" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
python3 $L/run.py train S1 > $L/train-S1.out 2>&1
python3 $L/run.py train S3 > $L/train-S3.out 2>&1
python3 $L/run.py compose > $L/compose.out 2>&1
git add $L && git commit -q -m "Freeze V31 trained models before any evaluation" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
python3 $L/run.py eval c1 > $L/eval-c1.log 2>&1
python3 $L/run.py eval c3 > $L/eval-c3.log 2>&1
python3 $L/run.py eval games > $L/eval-games.log 2>&1
python3 $L/analyze.py > /dev/null
echo done > $L/pipeline.done
