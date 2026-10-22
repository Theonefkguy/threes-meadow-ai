#!/bin/bash
# V50 part 3 (lq job, resumable): PPO from the best DAgger model (highest mean score in the 500-game
# test among dag6-dag10), 80 iterations; then the same tests as before plus the late-start test with
# each start played 5 times (play.py late ... 4), for the starting model too.
set -e
cd "$(dirname "$0")"
init=$(uv run python -c "
import json; from pathlib import Path
r = {n: json.loads(Path(f'results/play-{n}.json').read_text())['meanScore'] for n in ['dag6', 'dag7', 'dag8', 'dag9', 'dag10'] if Path(f'results/play-{n}.json').exists()}
print(max(r, key=r.get))")
echo "start from $init"; echo "$init" > results/rl1-init.txt
[ -f results/late5-$init.json ] || uv run python play.py runs/$init/model.pt late results/late5-$init.json 4
[ -f runs/rl1/model.pt ] || uv run python rl.py rl1 runs/$init/model.pt 80
[ -f results/play-rl1.json ] || uv run python play.py runs/rl1/model.pt 500 results/play-rl1.json
[ -f results/late5-rl1.json ] || uv run python play.py runs/rl1/model.pt late results/late5-rl1.json 4
