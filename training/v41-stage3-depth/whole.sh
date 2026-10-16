#!/bin/bash
# V41 step 3 (repository root): 4096 paired whole games, shipped float16 weights (native copies),
# cycle rule, cycle-aware website search; arm W keeps 5-ply / 0.01 after 6144, arm D switches to
# 6-ply / 0.005 (the step-2 choice). Both arms save first-6144 checkpoints. whole.sh W|D
set -e
L=training/v41-stage3-depth; Q=training/web-release/models; ARM=$1
case $ARM in W) LD=5; LT=0.01 ;; D) LD=6; LT=0.005 ;; *) exit 2 ;; esac
python3 $L/check.py
mkdir -p $L/whole $L/snaps
export L Q ARM LD LT
seq -f %.0f 41100001 16 41104096 | xargs -P 8 -n 1 sh -c '
  out=$L/whole/$ARM-$0.jsonl; [ -s $out ] && [ $(wc -l < $out) -eq 16 ] && exit 0
  rm -f $out.part $L/snaps/$ARM-$0.part
  $L/bin/game $Q/ntuple-v31.bin $Q/v31-s2q.res,$Q/v31-s1q.res,$Q/v31-s3q.res $LD $LT $0 16 $out.part $L/snaps/$ARM-$0.part && mv $out.part $out && mv $L/snaps/$ARM-$0.part $L/snaps/$ARM-$0.txt'
