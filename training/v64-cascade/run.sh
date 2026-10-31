#!/bin/bash
# V64 (repository root): every search iteration on the 200,000 sampled positions, chunks of 5000, 12 at a time;
# finished chunks are skipped, so the script can be rerun.
# (build: g++ -std=c++17 -O3 -march=native -o training/v64-cascade/bin/cascade training/v64-cascade/cascade.cpp)
set -e
L=training/v64-cascade; Q=training/web-release/models; D="/Volumes/Extreme SSD/Datasets/threes-v64"
mkdir -p "$D/runs"
export L Q D
seq -f %.0f 0 5000 199999 | xargs -P 12 -n 1 sh -c '
  s=$0; out="$D/runs/c-$s.txt"
  [ -s "$out" ] && [ $(wc -l < "$out") -eq 5000 ] && exit 0
  $L/bin/cascade $Q/ntuple-v31.bin $Q/v31-s2q.res,$Q/v31-s1q.res,$Q/v31-s3q.res "$D/positions.txt" $s 5000 "$out.part" && mv "$out.part" "$out"'
