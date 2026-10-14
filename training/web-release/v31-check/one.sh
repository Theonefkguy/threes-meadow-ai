#!/bin/bash
# One 16-game chunk of check 1 starting at seed $1 (called by run.sh).
set -e
D=training/web-release/v31-check; M=/Users/luipy/Projects/active/threes-meadow-ai/training; Q=training/web-release/models
[ -s $D/games/q-$1.jsonl ] && exit 0
rm -f $D/games/q-$1.part
CLAMP=1 $M/v31-stages/bin/game $Q/W31.bin $Q/v31-s2q.res,$Q/v31-s1q.res,$Q/v31-s3q.res 5 0.01 0 1 $1 16 $D/games/q-$1.part
mv $D/games/q-$1.part $D/games/q-$1.jsonl
