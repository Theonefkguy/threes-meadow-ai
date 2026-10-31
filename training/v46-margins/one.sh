#!/bin/bash
# V46: 4 games from seed $1 (lq chunked job).
set -e
L=training/v46-margins; Q=training/web-release/models
out=$L/data/m-$1.jsonl; gout=$L/data/g-$1.jsonl; [ -s $gout ] && [ $(wc -l < $gout) -eq 4 ] && exit 0
mkdir -p $L/data; rm -f $out.part $gout.part
$L/bin/margins $Q/ntuple-v31.bin $Q/v31-s2q.res,$Q/v31-s1q.res,$Q/v31-s3q.res $1 4 $out.part $gout.part
mv $out.part $out; mv $gout.part $gout
