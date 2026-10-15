#!/bin/bash
# V35 web release check (repository root): the shipped float16 V31 weights under the bonus-cycle
# rule with the cycle-aware website search, on the V31 release seeds.
set -e
D=training/web-release/v35-check; Q=training/web-release/models
python3 $D/check.py
mkdir -p $D/games
export D Q
seq -f %.0f 31300001 16 31302048 | xargs -P 8 -n 1 sh -c '
  out=$D/games/c-$0.jsonl; [ -s $out ] && [ $(wc -l < $out) -eq 16 ] && exit 0
  rm -f $out.part
  CLAMP=1 training/v35-cycle/bin/game $Q/ntuple-v31.bin $Q/v31-s2q.res,$Q/v31-s1q.res,$Q/v31-s3q.res 5 0.01 1 1 $0 16 $out.part && mv $out.part $out'
