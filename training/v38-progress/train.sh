#!/bin/bash
# Step 2: train arms. train.sh PC -> P (joint, with bias) and C (joint, no bias) at the same time,
# 8 threads each, 10800 s; train.sh B -> bias only, 8 threads, 3600 s. Strong player: complete
# 3-ply cycle-aware search (clamped).
set -e
. training/v38-progress/common.sh
python3 $L/check.py
mkdir -p $L/models
run(){ [ -s $L/models/$1-final.bin ] && return 0; $L/bin/train $W31 $RES $2 $3 8 $4 $5 $L/train-$1.log.jsonl 3 0 0.5 $L/models/$1 $TRAINPOOL; }
case $1 in
 PC) run P zero joint 10800 38210001 & run C - joint 10800 38220001 & wait ;;
 B) run B zero bias 3600 38230001 ;;
esac
for a in P C B; do [ -s $L/models/$a-final.bin ] || [ $1 != B ] || exit 1; done
