#!/bin/bash
# Check 1 (repository root of the web worktree): quantized V31 on the V31 whole-game seeds.
set -e
D=training/web-release/v31-check; mkdir -p $D/games
seq 31300001 16 31302048 | xargs -P 16 -n 1 bash $D/one.sh   # xargs -I hit the macOS command-length limit
python3 $D/analyze.py
