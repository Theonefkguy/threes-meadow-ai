"""V61 step 2 positions: for each game lost at stage 3072 or 6144 (top card at the last move >= 3072), one position
10-30 moves before the end (uniform offset, seed 61), as the public state the website AI sees.
  -> positions.txt (id board ncards cards bonus counts cycle), positions.json (id, seed, move, offset, top, empty)"""
import json
from pathlib import Path
import numpy as np
L = Path(__file__).resolve().parent; T = L.parent
DIRS = [T / 'v46-margins' / 'data', Path('/Volumes/Extreme SSD/Datasets/threes-v50/data')]
last = [r for r in json.loads((L / 'last.json').read_text()) if r['top'] >= 13]
rng = np.random.default_rng(61); want = {}
for r in sorted(last, key=lambda r: r['seed']):
    o = int(rng.integers(10, 31)); want[(r['seed'], r['move'] - o)] = o
found = {}
for D in DIRS:
    for f in sorted(D.glob('m-*.jsonl')):
        for line in f.open():
            m = json.loads(line)
            if (m['seed'], m['move']) in want: found[(m['seed'], m['move'])] = m
lines, meta = [], []
for i, key in enumerate(sorted(want)):
    m = found[key]; c = m['next']['cards']
    lines.append(' '.join(map(str, [i, *m['board'], len(c), *c, int(m['next']['bonus']), *m['counts'], *m['cycle']])))
    meta.append({'id': i, 'seed': key[0], 'move': key[1], 'offset': want[key], 'top': max(m['board']), 'empty': m['board'].count(0)})
(L / 'positions.txt').write_text('\n'.join(lines) + '\n'); (L / 'positions.json').write_text(json.dumps(meta) + '\n')
print(len(lines), 'positions; top', np.bincount([x['top'] for x in meta])[11:], 'empty', np.bincount([x['empty'] for x in meta]))
