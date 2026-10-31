#!/bin/bash
# V68 (repository root): bench on all 6000 V62 positions, single process, then analysis.
set -e
L=training/v68-profile; Q=training/web-release/models
$L/bin/bench $Q/ntuple-v31.bin $Q/v31-s2q.res,$Q/v31-s1q.res,$Q/v31-s3q.res training/v62-time-share/positions.txt 0 6000 $L/bench.txt
