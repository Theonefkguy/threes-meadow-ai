#!/bin/bash
# Step 1: cycle-rule first-6144 pool, seeds 38100001-38103400 (25-game chunks), merged by seed.
set -e
. training/v38-progress/common.sh
python3 $L/check.py
mkdir -p $L/pool
export L W31 RES
seq -f %.0f 38100001 25 38103400 | xargs -P 12 -n 1 sh -c '
  out=$L/pool/s-$0; [ -s $out.log ] && [ $(wc -l < $out.log) -eq 25 ] && exit 0
  rm -f $out.part $out.log.part
  $L/bin/pool $W31 $RES $0 25 $out.part $out.log.part && mv $out.part $out.txt && mv $out.log.part $out.log'
python3 - <<'PY'
import glob, json
L = 'training/v38-progress'
rows = sorted((l for f in glob.glob(f'{L}/pool/s-*.txt') for l in open(f)), key=lambda l: int(l.split()[0]))
logs = [json.loads(l) for f in glob.glob(f'{L}/pool/s-*.log') for l in open(f)]
assert len(logs) == 3400
print(json.dumps({'games': len(logs), 'reached6144': sum(r['reached'] for r in logs), 'kept': len(rows)}))
assert len(rows) >= 2048, 'fewer than 2048 states: extend per protocol'
open(f'{L}/pool/first6144.txt', 'w').write(''.join(rows[:2048]))
PY
