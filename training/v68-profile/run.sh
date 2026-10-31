#!/bin/bash
# V68 (repository root): whole-game reproduction check of the faster exact search. The 256 seeds of V66, arms D6
# (original search and model, depth 6 only = V66 arm D6) and F (searcht.h + FastModel<true>: compile-time maps, float16
# residuals, fast moves / key / top card) in the same job for timing under the same load; chunks of 8 games, 12 at a
# time; finished chunks are skipped.
# (build: g++ -std=c++17 -O3 -march=native -o training/v68-profile/bin/game training/v68-profile/game.cpp)
set -e
L=training/v68-profile; Q=training/web-release/models
mkdir -p $L/games $L/snaps
export L Q
for c in $(seq -f %.0f 41100001 256 41104096); do for s in $c $((c + 8)); do echo D6 $s; echo F $s; done; done | xargs -P 12 -n 2 sh -c '
  a=$0; s=$1; out=$L/games/$a-$s.jsonl
  [ -s $out ] && [ $(wc -l < $out) -eq 8 ] && exit 0
  rm -f $out.part $L/snaps/$a-$s.part
  if [ $a = D6 ]; then arm=0; else arm=2; fi
  $L/bin/game $Q/ntuple-v31.bin $Q/v31-s2q.res,$Q/v31-s1q.res,$Q/v31-s3q.res $arm $s 8 $out.part $L/snaps/$a-$s.part
  mv $out.part $out; mv $L/snaps/$a-$s.part $L/snaps/$a-$s.txt'
python3.12 $L/games.py
