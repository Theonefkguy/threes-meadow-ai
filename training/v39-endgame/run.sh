#!/bin/bash
# V39 (repository root): replay the V38 W31 arm with milestone tracking, check it, analyze.
set -e
L=training/v39-endgame; M=training/v31-stages/models
mkdir -p $L/games
export L M
seq -f %.0f 0 16 2047 | xargs -P 8 -n 1 sh -c '
  out=$L/games/d-$0.jsonl; [ -s $out ] && [ $(wc -l < $out) -eq 16 ] && exit 0
  rm -f $out.part
  $L/bin/diag $M/W31.bin training/v30-parallel/models/P-final.res,$M/S1-final.res,$M/S3-final.res training/v38-progress/pool/first6144.txt $0 16 38400001 $out.part && mv $out.part $out'
python3 $L/analyze.py
