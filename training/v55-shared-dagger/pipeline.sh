#!/bin/bash
# V55 (lq job, resumable): DAgger rounds with the V54 shared model as the student, from round 15 on,
# until eval.py's stopping rule says stop. Each round: shared.py dagger (400 games + 800 late starts,
# labelled by the website AI), fit.py (fine-tune on everything so far), eval.py. Every step is skipped
# when its output exists; fit.py resumes from its checkpoint.
set -e
cd "$(dirname "$0")"
D="/Volumes/Extreme SSD/Datasets/threes-v50"
PY="uv run --project ../v50-transformer python"
prev=../v54-symmetry/runs/shared/model.pt
[ "$($PY eval.py 14 | tail -1)" = go ] || exit 0
for r in $(seq 15 24); do
  $PY shared.py dagger $r $prev
  [ -f runs/r$r/model.pt ] || $PY fit.py $r $prev
  v=$($PY eval.py $r | tail -1); echo "round $r: $v"
  prev=runs/r$r/model.pt
  [ "$v" = go ] || break
done
