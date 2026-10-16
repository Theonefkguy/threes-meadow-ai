#!/bin/bash
# V41 step 2 (repository root): arms D6 (6-ply / 0.01) and T5 (5-ply / 0.005) from the first 6144 on the
# V38 pool and rollout seeds (V40 deep tool, TRIGGER start). Baseline = V39 replay, S = V40 arm S.
set -e
L=training/v41-stage3-depth; V=training/v40-late-depth; M=training/v31-stages/models
python3 $L/check.py
mkdir -p $L/games
export L V M
for a in D6 T5; do seq -f %.0f 0 16 2047 | sed "s/^/$a /"; done | xargs -P 8 -n 2 sh -c '
  a=$0; s=$1; out=$L/games/$a-$s.jsonl; [ -s $out ] && [ $(wc -l < $out) -eq 16 ] && exit 0
  case $a in D6) d=6; t=0.01 ;; T5) d=5; t=0.005 ;; esac
  rm -f $out.part
  $V/bin/deep $M/W31.bin training/v30-parallel/models/P-final.res,$M/S1-final.res,$M/S3-final.res training/v38-progress/pool/first6144.txt $s 16 38400001 start $d $t $out.part && mv $out.part $out'
python3 $L/analyze.py
