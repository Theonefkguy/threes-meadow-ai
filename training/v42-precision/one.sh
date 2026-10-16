#!/bin/bash
# V42: one 16-game chunk of the full-precision arm, seeds $1..$1+15 (called by lq as a chunked job).
set -e
L=training/v42-precision; V=training/v41-stage3-depth; M=training/v31-stages/models
out=$L/games/F-$1.jsonl; [ -s $out ] && [ $(wc -l < $out) -eq 16 ] && exit 0
mkdir -p $L/games $L/snaps; rm -f $out.part $L/snaps/F-$1.part
$V/bin/game $M/W31.bin training/v30-parallel/models/P-final.res,$M/S1-final.res,$M/S3-final.res 6 0.005 $1 16 $out.part $L/snaps/F-$1.part
mv $out.part $out; mv $L/snaps/F-$1.part $L/snaps/F-$1.txt
