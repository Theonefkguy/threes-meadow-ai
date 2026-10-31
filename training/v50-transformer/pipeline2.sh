#!/bin/bash
# V50 part 2 (lq job, resumable): DAgger rounds FIRST-LAST (default 7-10) with more late-game positions.
# Each round: the student plays 400 games from the start plus 800 late starts (400 from where a training
# AI game first had 1536, 400 from 3072); every position is labelled by the AI. Fine-tune on everything
# so far (all rounds repeated 4 times; round 7+ positions with a card >= 1536 repeated 2 more times).
# After each model: 500 whole games (play.py) and the late-start test on the validation AI games
# (from round 11 on also the larger test, each start played 5 times: results/late5-*.json).
set -e
cd "$(dirname "$0")"
D="/Volumes/Extreme SSD/Datasets/threes-v50"
FIRST=${FIRST:-7}; LAST=${LAST:-10}
extra=$(for r in 1 2 3 4 5 6; do printf "%s," "$D/dagger/round-$r.npz"; done); extra=${extra%,}
late=$(for r in $(seq 7 $((FIRST - 1))); do printf "%s," "$D/dagger/round-$r.npz"; done); late=${late%,}
[ -f results/late-dag6.json ] || uv run python play.py runs/dag6/model.pt late results/late-dag6.json
prev=dag$((FIRST - 1))
for r in $(seq $FIRST $LAST); do
  uv run python dagger.py $r runs/$prev/model.pt 400 12 800
  late="${late:+$late,}$D/dagger/round-$r.npz"
  [ -f runs/dag$r/model.pt ] || uv run python train.py dag$r --init runs/$prev/model.pt --extra "$extra" --late-extra "$late" --repeat 4 --late 2 --epochs 2 --lr 3e-4
  [ -f results/play-dag$r.json ] || uv run python play.py runs/dag$r/model.pt 500 results/play-dag$r.json
  [ -f results/late-dag$r.json ] || uv run python play.py runs/dag$r/model.pt late results/late-dag$r.json
  [ $r -lt 11 ] || [ -f results/late5-dag$r.json ] || uv run python play.py runs/dag$r/model.pt late results/late5-dag$r.json 4
  prev=dag$r
done
