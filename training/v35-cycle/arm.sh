#!/bin/bash
# V35: one arm's whole games (repository root). arm.sh U|A   (U: search unaware of the cycle, A: aware)
set -e
L=training/v35-cycle; M=training/v31-stages/models; ARM=$1
case $ARM in U) AWARE=0 ;; A) AWARE=1 ;; *) echo "unknown arm"; exit 2 ;; esac
python3 $L/check.py
mkdir -p $L/games
export ARM AWARE L M
seq -f %.0f 35100001 16 35104096 | xargs -P 8 -n 1 sh -c '
  out=$L/games/$ARM-$0.jsonl; [ -s $out ] && [ $(wc -l < $out) -eq 16 ] && exit 0
  rm -f $out.part
  CLAMP=1 $L/bin/game $M/W31.bin training/v30-parallel/models/P-final.res,$M/S1-final.res,$M/S3-final.res 5 0.01 1 $AWARE $0 16 $out.part && mv $out.part $out'
