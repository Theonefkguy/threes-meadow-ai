#!/bin/bash
# V30 full pipeline, run detached from the repository root: nohup caffeinate -i bash training/v30-parallel/pipeline.sh &
set -e
L=training/v30-parallel
python3 $L/run.py train S > $L/train-S.out 2>&1
python3 $L/run.py train P > $L/train-P.out 2>&1
python3 $L/run.py freeze-models > $L/freeze.out 2>&1
git add $L/models-manifest.json $L/train-S.log.jsonl $L/train-P.log.jsonl $L/train-S.out $L/train-P.out
git commit -q -m "Freeze V30 trained models before any evaluation" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
python3 $L/run.py select > $L/select.log 2>&1
python3 $L/analyze.py select > /dev/null
python3 $L/run.py games > $L/games.log 2>&1
python3 $L/analyze.py games > /dev/null
echo done > $L/pipeline.done
