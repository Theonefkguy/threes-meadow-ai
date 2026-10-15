#!/bin/bash
# Step 3: freeze pool and models, then continue every pool state with W31, B, C and P
# (cycle-aware website search) to 12288 or death, rollout seeds 38400001+index; then analyze.
set -e
. training/v38-progress/common.sh
python3 $L/check.py
python3 $L/freeze.py
mkdir -p $L/eval
export L W31 RES M
jobs(){ for a in W31 B C P; do seq -f %.0f 0 16 2047 | sed "s/^/$a /"; done; }
jobs | xargs -P 16 -n 2 sh -c '
  a=$0; s=$1; out=$L/eval/$a-$s.jsonl; [ -s $out ] && [ $(wc -l < $out) -eq 16 ] && exit 0
  case $a in W31) b=$W31; r=$RES; bias=- ;; C) b=$L/models/C-final.bin; r=training/v30-parallel/models/P-final.res,$M/S1-final.res,$L/models/C-final.res; bias=- ;;
   *) b=$L/models/$a-final.bin; r=training/v30-parallel/models/P-final.res,$M/S1-final.res,$L/models/$a-final.res; bias=$L/models/$a-final.bias ;; esac
  rm -f $out.part; $L/bin/cont $b $r $bias $L/pool/first6144.txt $s 16 38400001 $out.part && mv $out.part $out'
python3 $L/analyze.py
