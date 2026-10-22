#!/bin/bash
# V50 part 2 (lq job, resumable): DAgger rounds 7-10 from dag6 with more late-game positions.
# Each round: the student plays 400 games from the start plus 800 late starts (400 from where a training
# AI game first had 1536, 400 from 3072); every position is labelled by the AI. Fine-tune on everything
# so far (all rounds repeated 4 times; round 7+ positions with a card >= 1536 repeated 2 more times).
# After each model: 500 whole games (play.py) and the late-start test on the validation AI games.
set -e
cd "$(dirname "$0")"
D="/Volumes/Extreme SSD/Datasets/threes-v50"
extra=$(for r in 1 2 3 4 5 6; do printf "%s," "$D/dagger/round-$r.npz"; done); extra=${extra%,}; late=""
[ -f results/late-dag6.json ] || uv run python play.py runs/dag6/model.pt late results/late-dag6.json
prev=dag6
for r in 7 8 9 10; do
  uv run python dagger.py $r runs/$prev/model.pt 400 12 800
  late="${late:+$late,}$D/dagger/round-$r.npz"
  [ -f runs/dag$r/model.pt ] || uv run python train.py dag$r --init runs/$prev/model.pt --extra "$extra" --late-extra "$late" --repeat 4 --late 2 --epochs 2 --lr 3e-4
  [ -f results/play-dag$r.json ] || uv run python play.py runs/dag$r/model.pt 500 results/play-dag$r.json
  [ -f results/late-dag$r.json ] || uv run python play.py runs/dag$r/model.pt late results/late-dag$r.json
  prev=dag$r
done
