#!/bin/bash
# Replay the 2048 release-check games with combo tracking (training/v35-cycle/combo.cpp), then
# check every replayed game against the recorded one and summarise (combo.json).
set -e
D=training/web-release/v35-check; Q=training/web-release/models
mkdir -p $D/combo
export D Q
seq -f %.0f 31300001 16 31302048 | xargs -P 8 -n 1 sh -c '
  out=$D/combo/c-$0.jsonl; [ -s $out ] && [ $(wc -l < $out) -eq 16 ] && exit 0
  rm -f $out.part
  CLAMP=1 training/v35-cycle/bin/combo $Q/ntuple-v31.bin $Q/v31-s2q.res,$Q/v31-s1q.res,$Q/v31-s3q.res 5 0.01 1 1 $0 16 $out.part && mv $out.part $out'
python3 - <<'PY'
import json, glob, math
D = 'training/web-release/v35-check'
load = lambda pat: {r['seed']: r for f in glob.glob(pat) for r in map(json.loads, open(f))}
old, new = load(f'{D}/games/c-*.jsonl'), load(f'{D}/combo/c-*.jsonl')
assert len(old) == len(new) == 2048
for s in old:
    assert all(old[s][k] == new[s][k] for k in ('r1536', 'r3072', 'r6144', 'r12288', 'turn6144', 'moves', 'score', 'bonuses')), s
n = len(new); Z = 1.959963984540054
def wilson(k):
    p = k / n; d = 1 + Z * Z / n; c = p + Z * Z / (2 * n); h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)); return [round((c - h) / d, 4), round((c + h) / d, 4)]
rows = {'1536': 'r1536', '3072': 'r3072', '6144': 'r6144', '6144+3072': 'c2', '6144+3072+1536': 'c3', '6144+3072+1536+768': 'c4', '12288': 'r12288'}
out = {'games': n, 'replayIdentical': True}
for name, k in rows.items():
    c = sum((r[k] >= 0) if k.startswith('c') else r[k] for r in new.values())
    out[name] = {'count': c, 'rate': c / n, 'wilson95': wilson(c)}
out['12288 among games with 6144+3072+1536+768'] = sum(r['r12288'] for r in new.values() if r['c4'] >= 0)
out['12288 among games with 6144+3072'] = sum(r['r12288'] for r in new.values() if r['c2'] >= 0)
open(f'{D}/combo.json', 'w').write(json.dumps(out, indent=1) + '\n'); print(json.dumps(out, indent=1))
PY
