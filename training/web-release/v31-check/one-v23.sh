#!/bin/bash
# One 16-game chunk of check 4 (V23 on the V31 seeds) starting at seed $1.
set -e
D=training/web-release/v31-check; M=/Users/luipy/Projects/active/threes-meadow-ai/training
[ -s $D/games/v23-$1.jsonl ] && exit 0
rm -f $D/games/v23-$1.part
CLAMP=1 $M/v31-stages/bin/game dist/models/ntuple-v23.bin - 5 0.01 0 1 $1 16 $D/games/v23-$1.part
mv $D/games/v23-$1.part $D/games/v23-$1.jsonl
