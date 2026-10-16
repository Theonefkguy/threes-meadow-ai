#!/bin/bash
# V40 (repository root): arms L (deep from the final-768 milestone) and S (deep from the first 6144),
# 6-ply / cutoff 0.005, on the V38 pool with the V38 rollout seeds; baseline = V39 replay (= V38 W31).
set -e
L=training/v40-late-depth; M=training/v31-stages/models
python3 $L/check.py
mkdir -p $L/games
export L M
for a in L S; do seq -f %.0f 0 16 2047 | sed "s/^/$a /"; done | xargs -P 8 -n 2 sh -c '
  a=$0; s=$1; t=$([ $a = L ] && echo m768 || echo start); out=$L/games/$a-$s.jsonl
  [ -s $out ] && [ $(wc -l < $out) -eq 16 ] && exit 0
  rm -f $out.part
  $L/bin/deep $M/W31.bin training/v30-parallel/models/P-final.res,$M/S1-final.res,$M/S3-final.res training/v38-progress/pool/first6144.txt $s 16 38400001 $t 6 0.005 $out.part && mv $out.part $out'
python3 $L/analyze.py
