#!/bin/bash
# V43: one 16-game chunk of arm E (6-ply / 0.005 from the first move), seeds $1..$1+15 (lq chunked job).
set -e
L=training/v43-early-depth; Q=training/web-release/models
out=$L/games/E-$1.jsonl; [ -s $out ] && [ $(wc -l < $out) -eq 16 ] && exit 0
mkdir -p $L/games $L/snaps; rm -f $out.part $L/snaps/E-$1.part
$L/bin/game $Q/ntuple-v31.bin $Q/v31-s2q.res,$Q/v31-s1q.res,$Q/v31-s3q.res 6 0.005 6 0.005 $1 16 $out.part $L/snaps/E-$1.part
mv $out.part $out; mv $L/snaps/E-$1.part $L/snaps/E-$1.txt
