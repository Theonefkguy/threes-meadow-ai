#!/bin/bash
# V34: one arm's 2048 whole games on the V31 release seeds (repository root). arm.sh C1|C2|C3
set -e
L=training/v34-lossy; M=training/web-release/models; O=$L/models; ARM=$1
case $ARM in
  C1) BASE=$O/v31-base16.bin; RES=$M/v31-s1q.res,$M/v31-s2q.res,$M/v31-s3q.res ;;
  C2) BASE=$M/ntuple-v31.bin; RES=$O/v31-s1-p8.res,$O/v31-s2-p8.res,$O/v31-s3-p8.res ;;
  C3) BASE=$O/v31-base16.bin; RES=$O/v31-s1-p8.res,$O/v31-s2-p8.res,$O/v31-s3-p8.res ;;
  *) echo "unknown arm"; exit 2 ;;
esac
python3 $L/check.py $ARM
mkdir -p $L/games
export ARM BASE RES L
seq -f %.0f 31300001 16 31302048 | xargs -P 8 -n 1 sh -c '
  out=$L/games/$ARM-$0.jsonl; [ -s $out ] && exit 0
  rm -f $out.part
  CLAMP=1 training/v31-stages/bin/game $BASE $RES 5 0.01 0 1 $0 16 $out.part && mv $out.part $out'
