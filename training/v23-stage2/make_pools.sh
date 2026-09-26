#!/bin/bash
# V23 pools (run from repo root). Official-modern rules.
#  onpolicy: states every 25 moves (max card 3072, alive) along 5-ply/0.01 clamped play of the
#            current model from the V21 train-3072 snapshots (rollout seeds 23100001+index).
#  eval: first-3072 snapshots from new openings 23110001+ (complete 3-ply). The first 512 by
#        seed form the screening pool, the next 2560 the confirmation pool.
set -e
P=training/v23-stage2/pools
if [ "$1" = onpolicy ]; then
  CLAMP=1 EVERY=25 /tmp/v21-snap sample training/v7/base.ntd 5 0.01 training/v21-late-train/pools/train-3072.txt 0 781 23100001 14 $P/onpolicy.txt
fi
if [ "$1" = eval ]; then
  /tmp/v21-snap pool 23110001 5000 $P/eval-1536.txt $P/eval-3072.raw
fi
