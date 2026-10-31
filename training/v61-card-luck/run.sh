#!/bin/bash
# V61 step 2 (repository root): 256 replays of 30 moves from each of the 860 positions, website AI 6-ply / 0.005.
# Chunks of 20 positions, 12 at a time; finished chunks are skipped.
set -e
L=training/v61-card-luck; Q=training/web-release/models
mkdir -p $L/rolls
export L Q
seq 0 20 859 | xargs -P 12 -n 1 sh -c '
  s=$0; out=$L/rolls/r-$s.jsonl
  [ -s $out ] && [ $(wc -l < $out) -eq 5120 ] && exit 0
  rm -f $out.part
  $L/bin/roll $Q/ntuple-v31.bin $Q/v31-s2q.res,$Q/v31-s1q.res,$Q/v31-s3q.res $L/positions.txt $s 20 256 30 $out.part && mv $out.part $out'
