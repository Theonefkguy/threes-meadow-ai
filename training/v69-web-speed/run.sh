#!/bin/bash
# V69 whole games: 96 seeds (1001-1096) in 12 processes of 8 games each; output in games/.
set -e
cd "$(dirname "$0")"; mkdir -p games
for p in $(seq 0 11); do
  first=$((1001 + p*8)); rm -f games/g$p.jsonl
  node games.mjs ref,new,new:1.2.3.4.6 $first 8 games/g$p.jsonl &
done
wait
