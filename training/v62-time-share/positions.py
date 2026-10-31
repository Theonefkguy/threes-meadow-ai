"""V62 positions: 6000 moves sampled uniformly from all moves of the 800 V50 website-AI games (seed 62), as the
public state the AI sees.  -> positions.txt (V61 format), positions.json (id, seed, move, top, empty)"""
import json
from pathlib import Path
import numpy as np
L = Path(__file__).resolve().parent
files = sorted(Path('/Volumes/Extreme SSD/Datasets/threes-v50/data').glob('m-*.jsonl'))
sizes = [sum(1 for _ in f.open()) for f in files]; N = sum(sizes)
pick = set(np.random.default_rng(62).choice(N, 6000, replace=False).tolist())
lines, meta, k = [], [], 0
for f in files:
    for line in f.open():
        if k in pick:
            m = json.loads(line); c = m['next']['cards']; i = len(lines)
            lines.append(' '.join(map(str, [i, *m['board'], len(c), *c, int(m['next']['bonus']), *m['counts'], *m['cycle']])))
            meta.append({'id': i, 'seed': m['seed'], 'move': m['move'], 'top': max(m['board']), 'empty': m['board'].count(0)})
        k += 1
(L / 'positions.txt').write_text('\n'.join(lines) + '\n'); (L / 'positions.json').write_text(json.dumps(meta) + '\n')
print(N, 'moves;', len(lines), 'positions; top', np.bincount([x['top'] for x in meta]), 'empty', np.bincount([x['empty'] for x in meta]))
