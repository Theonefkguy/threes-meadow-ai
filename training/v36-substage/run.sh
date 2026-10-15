#!/bin/bash
# V36 sub-stage screen (repository root): five fits in parallel on the V33 records, then analyze.
set -e
L=training/v36-substage; D=training/v33-patterns/data
python3 $L/check.py
P5="0,1,2,3,4;4,5,6,7,8;0,1,2,4,5;4,5,6,8,9"
for s in none two three four bias3; do
  [ -s $L/fit-$s.json ] || $L/bin/fit $s dist/models/ntuple-v23.bin "$P5" $D/*.bin -- $L/fit-$s.json &
done
wait
python3 $L/analyze.py
