#!/bin/bash
# V50 pipeline (lq job, resumable): pack the 1000 AI games, train the base transformer, then DAgger
# rounds (student plays 400 games, the AI labels every position, fine-tune on everything so far).
# After each model: 500 whole games with no search (play.py, seeds 50900001..).
set -e
cd "$(dirname "$0")"
D="/Volumes/Extreme SSD/Datasets/threes-v50"; R=${ROUNDS:-6}
[ -f "$D/packed.ok" ] || { uv run python pack.py; touch "$D/packed.ok"; }
[ -f runs/base/model.pt ] || uv run python train.py base --epochs 12
[ -f results/play-base.json ] || uv run python play.py runs/base/model.pt 500 results/play-base.json
prev=base; extra=""
for r in $(seq 1 $R); do
  uv run python dagger.py $r runs/$prev/model.pt 400 12
  extra="${extra:+$extra,}$D/dagger/round-$r.npz"
  [ -f runs/dag$r/model.pt ] || uv run python train.py dag$r --init runs/$prev/model.pt --extra "$extra" --repeat 4 --epochs 2 --lr 3e-4
  [ -f results/play-dag$r.json ] || uv run python play.py runs/dag$r/model.pt 500 results/play-dag$r.json
  prev=dag$r
done
