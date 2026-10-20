#!/bin/bash
# V45 scan: 16 pool states from index $1 (lq chunked job).
set -e
L=training/v45-explain; Q=training/web-release/models
out=$L/scan/m-$1.jsonl; gout=$L/scan/g-$1.jsonl; [ -s $gout ] && [ $(wc -l < $gout) -eq $(( $1 + 16 > 2783 ? 2783 - $1 : 16 )) ] && exit 0
mkdir -p $L/scan; rm -f $out.part $gout.part
$L/bin/scan $Q/ntuple-v31.bin $Q/v31-s2q.res,$Q/v31-s1q.res,$Q/v31-s3q.res $L/pool-d6144.txt $1 16 45100001 $out.part $gout.part
mv $out.part $out; mv $gout.part $gout
