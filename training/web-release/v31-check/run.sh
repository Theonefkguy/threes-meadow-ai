#!/bin/bash
# Check 1 (repository root of the web worktree): quantized V31 on the V31 whole-game seeds.
set -e
D=training/web-release/v31-check; M=/Users/luipy/Projects/active/threes-meadow-ai/training; Q=training/web-release/models
mkdir -p $D/games
seq 31300001 16 31302048 | xargs -P 16 -I{} sh -c "[ -s $D/games/q-{}.jsonl ] || (CLAMP=1 $M/v31-stages/bin/game $Q/W31.bin $Q/v31-s2q.res,$Q/v31-s1q.res,$Q/v31-s3q.res 5 0.01 0 1 {} 16 $D/games/q-{}.part && mv $D/games/q-{}.part $D/games/q-{}.jsonl)"
python3 $D/analyze.py
