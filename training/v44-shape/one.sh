#!/bin/bash
# V44: one chunk (comma-separated pool indices in $1) of the shape replay (lq chunked job).
set -e
L=training/v44-shape; M=training/v31-stages/models; k=${1%%,*}
out=$L/games/r-$k.jsonl; [ -s $out ] && exit 0
mkdir -p $L/games; rm -f $out.part $L/games/s-$k.part
$L/bin/shape $M/W31.bin training/v30-parallel/models/P-final.res,$M/S1-final.res,$M/S3-final.res training/v38-progress/pool/first6144.txt $1 38400001 $out.part $L/games/s-$k.part
mv $L/games/s-$k.part $L/games/s-$k.jsonl; mv $out.part $out
