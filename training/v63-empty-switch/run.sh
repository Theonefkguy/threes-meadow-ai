#!/bin/bash
# V63 (repository root): whole games, seeds 41100001-41104096 (V43), arm K3 = 6-ply / 0.005 but 5-ply / 0.01 when the
# board has >= 3 empty cells; control = V43 arm E records. Every 16th chunk also reruns arm E (EMPTY_K 17) in the same
# job for timing under the same load (and as a reproduction check). Chunks of 16 games, 12 at a time; finished chunks
# are skipped, so the script can be rerun.
# (build: g++ -std=c++17 -O3 -march=native -o training/v63-empty-switch/bin/game training/v63-empty-switch/game.cpp)
set -e
L=training/v63-empty-switch; Q=training/web-release/models
mkdir -p $L/games $L/snaps
export L Q
for s in $(seq -f %.0f 41100001 16 41104096); do echo K3 $s; [ $(( (s - 41100001) / 16 % 16 )) -eq 0 ] && echo E $s; done | xargs -P 12 -n 2 sh -c '
  a=$0; s=$1; out=$L/games/$a-$s.jsonl
  [ -s $out ] && [ $(wc -l < $out) -eq 16 ] && exit 0
  rm -f $out.part $L/snaps/$a-$s.part
  if [ $a = K3 ]; then k=3; else k=17; fi
  $L/bin/game $Q/ntuple-v31.bin $Q/v31-s2q.res,$Q/v31-s1q.res,$Q/v31-s3q.res 5 0.01 $k $s 16 $out.part $L/snaps/$a-$s.part
  mv $out.part $out; mv $L/snaps/$a-$s.part $L/snaps/$a-$s.txt'
python3.12 $L/analyze.py
