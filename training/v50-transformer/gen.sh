#!/bin/bash
# V50: 4 website-AI games from seed $1 with per-direction root values (V46 margins tool), for distillation.
# Output on the external disk: $D/m-SEED.jsonl (moves), $D/g-SEED.jsonl (games). lq chunked job.
set -e
D="/Volumes/Extreme SSD/Datasets/threes-v50/data"; Q=training/web-release/models
out="$D/m-$1.jsonl"; gout="$D/g-$1.jsonl"; [ -s "$gout" ] && [ $(wc -l < "$gout") -eq 4 ] && exit 0
rm -f "$out.part" "$gout.part"
training/v46-margins/bin/margins $Q/ntuple-v31.bin $Q/v31-s2q.res,$Q/v31-s1q.res,$Q/v31-s3q.res $1 4 "$out.part" "$gout.part"
mv "$out.part" "$out"; mv "$gout.part" "$gout"
