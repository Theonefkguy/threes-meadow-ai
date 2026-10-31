#!/bin/bash
# V66 (repository root): whole-game reproduction check of the depth-6-only search. The 256 seeds of the V65 arm E
# reruns (chunks 41100001 + 256k, 16 games each), arms E (iterations from depth 1) and D6 (depth 6 only) in the same
# job for timing under the same load; chunks of 8 games, 12 at a time; finished chunks are skipped.
# (build: g++ -std=c++17 -O3 -march=native -o training/v66-direct-depth/bin/game training/v66-direct-depth/game.cpp)
set -e
L=training/v66-direct-depth; Q=training/web-release/models
mkdir -p $L/games $L/snaps
export L Q
for c in $(seq -f %.0f 41100001 256 41104096); do for s in $c $((c + 8)); do echo D6 $s; echo E $s; done; done | xargs -P 12 -n 2 sh -c '
  a=$0; s=$1; out=$L/games/$a-$s.jsonl
  [ -s $out ] && [ $(wc -l < $out) -eq 8 ] && exit 0
  rm -f $out.part $L/snaps/$a-$s.part
  if [ $a = D6 ]; then st=6; else st=1; fi
  $L/bin/game $Q/ntuple-v31.bin $Q/v31-s2q.res,$Q/v31-s1q.res,$Q/v31-s3q.res $st $s 8 $out.part $L/snaps/$a-$s.part
  mv $out.part $out; mv $L/snaps/$a-$s.part $L/snaps/$a-$s.txt'
python3.12 $L/games.py
