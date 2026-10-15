#!/bin/bash
# V37 progress-feature screen (repository root): five fits in parallel on the V33 records, then analyze.
set -e
L=training/v37-progress; D=training/v33-patterns/data
python3 $L/check.py
P5="0,1,2,3,4;4,5,6,7,8;0,1,2,4,5;4,5,6,8,9"
for s in none bias3 s2 s23 s234 chain mass; do
  [ -s $L/fit-$s.json ] || $L/bin/fit $s dist/models/ntuple-v23.bin "$P5" $D/*.bin -- $L/fit-$s.json &
done
wait
python3 $L/analyze.py
