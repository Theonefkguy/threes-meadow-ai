#!/bin/bash
# One collection chunk: pass $1 (A|B), start index $2, 100 starts.
set -e
L=training/v33-patterns; M=training/v31-stages/models
BASE=$([ "$1" = A ] && echo 33100001 || echo 33200001)
[ -s $L/data/$1-$2.bin ] && exit 0
rm -f $L/data/$1-$2.part
CLAMP=1 $L/bin/collect $M/W31.bin training/v30-parallel/models/P-final.res,$M/S1-final.res,$M/S3-final.res training/v27-stage3/pools/train-onpolicy.txt $2 100 $BASE 2 $L/data/$1-$2.part
mv $L/data/$1-$2.part $L/data/$1-$2.bin
