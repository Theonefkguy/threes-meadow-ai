#!/bin/bash
# V60 (repository root): arm D7F = 7-ply / cutoff 0.0025 every move from the first 6144, with the V59 program
# (build: g++ -std=c++17 -O3 -march=native -o training/v59-selective/bin/sel training/v59-selective/sel.cpp),
# on the V38 first-6144 pool with the V40 rollout seeds. Chunks of 16 states, 12 at a time; finished chunks are skipped.
set -e
L=training/v60-deep-cutoff; M=training/v31-stages/models
mkdir -p $L/games
export L M
seq -f %.0f 0 16 2047 | xargs -P 12 -n 1 sh -c '
  s=$0; out=$L/games/D7F-$s.jsonl
  [ -s $out ] && [ $(wc -l < $out) -eq 16 ] && exit 0
  rm -f $out.part
  training/v59-selective/bin/sel $M/W31.bin training/v30-parallel/models/P-final.res,$M/S1-final.res,$M/S3-final.res training/v38-progress/pool/first6144.txt $s 16 38400001 7 7 0.0025 0 $out.part && mv $out.part $out'
python3.12 $L/analyze.py
