#!/bin/bash
# V45 rollouts: 2 positions from job $1 (lq chunked job), 40 rollouts per arm.
set -e
L=training/v45-explain; Q=training/web-release/models
out=$L/rollouts/r-$1.jsonl; [ -s $out ] && [ $(wc -l < $out) -eq 160 ] && exit 0
mkdir -p $L/rollouts; rm -f $out.part
$L/bin/rollout $Q/ntuple-v31.bin $Q/v31-s2q.res,$Q/v31-s1q.res,$Q/v31-s3q.res $L/jobs.jsonl $1 2 40 45200001 $out.part
mv $out.part $out
