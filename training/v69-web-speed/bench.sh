#!/bin/bash
# V69 positions: all 6000 V62 positions, single process.
set -e
cd "$(dirname "$0")"
node bench.mjs ref,new,new:1.2.3.4.6,new:1.6 ../v62-time-share/positions.txt 0 6000 bench.txt
