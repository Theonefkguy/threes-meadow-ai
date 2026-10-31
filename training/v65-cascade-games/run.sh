#!/bin/bash
# V65 (repository root): whole games, seeds 41100001-41104096 (V43), arm C1 = the V64 cascade (judge.txt, 1% operating
# point); control = V43 arm E records. Every 16th chunk also reruns arm E (JUDGE -) in the same
# job for timing under the same load (and as a reproduction check). Chunks of 16 games, 12 at a time; finished chunks
# are skipped, so the script can be rerun.
# (build: g++ -std=c++17 -O3 -march=native -o training/v65-cascade-games/bin/game training/v65-cascade-games/game.cpp)
set -e
L=training/v65-cascade-games; Q=training/web-release/models
mkdir -p $L/games $L/snaps
export L Q
for s in $(seq -f %.0f 41100001 16 41104096); do echo C1 $s; [ $(( (s - 41100001) / 16 % 16 )) -eq 0 ] && echo E $s; done | xargs -P 12 -n 2 sh -c '
  a=$0; s=$1; out=$L/games/$a-$s.jsonl
  [ -s $out ] && [ $(wc -l < $out) -eq 16 ] && exit 0
  rm -f $out.part $L/snaps/$a-$s.part
  if [ $a = C1 ]; then j=training/v64-cascade/judge.txt; else j=-; fi
  $L/bin/game $Q/ntuple-v31.bin $Q/v31-s2q.res,$Q/v31-s1q.res,$Q/v31-s3q.res $j $s 16 $out.part $L/snaps/$a-$s.part
  mv $out.part $out; mv $L/snaps/$a-$s.part $L/snaps/$a-$s.txt'
python3.12 $L/analyze.py
