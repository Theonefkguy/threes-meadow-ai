#!/bin/bash
# V59 (repository root): arms D7 (7-ply every move) and SEL (6-ply, deepened to 7 when the root margin < 1%),
# cutoff 0.005, on the V38 first-6144 pool with the V40 rollout seeds; S6 = the V40 S arm records.
# Chunks of 16 states, D7 and SEL interleaved so both see the same machine load, 12 at a time; finished chunks are skipped, so the script can be rerun.
set -e
L=training/v59-selective; M=training/v31-stages/models
mkdir -p $L/games
export L M
for s in $(seq -f %.0f 0 16 2047); do echo D7 $s; echo SEL $s; done | xargs -P 12 -n 2 sh -c '
  a=$0; s=$1; out=$L/games/$a-$s.jsonl
  [ -s $out ] && [ $(wc -l < $out) -eq 16 ] && exit 0
  rm -f $out.part
  if [ $a = D7 ]; then set -- 7 7 0; else set -- 6 7 0.01; fi
  $L/bin/sel $M/W31.bin training/v30-parallel/models/P-final.res,$M/S1-final.res,$M/S3-final.res training/v38-progress/pool/first6144.txt $s 16 38400001 $1 $2 0.005 $3 $out.part && mv $out.part $out'
python3.12 $L/analyze.py
