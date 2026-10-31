"""V64 positions: 200,000 moves sampled uniformly from all moves of the 800 V50 website-AI games (seed 64), as the
public state the AI sees, plus the recorded move (a check that the depth-6 iteration is the website move).
  -> /Volumes/Extreme SSD/Datasets/threes-v64/positions.txt (V61 format), meta.npz (seed, move, top, empty, chosen)"""
import json
from pathlib import Path
import numpy as np
OUT = Path('/Volumes/Extreme SSD/Datasets/threes-v64')
files = sorted(Path('/Volumes/Extreme SSD/Datasets/threes-v50/data').glob('m-*.jsonl'))
sizes = [sum(1 for _ in f.open()) for f in files]; N = sum(sizes)
pick = set(np.random.default_rng(64).choice(N, 200_000, replace=False).tolist())
lines, meta, k = [], [], 0
for f in files:
    for line in f.open():
        if k in pick:
            m = json.loads(line); c = m['next']['cards']; i = len(lines)
            lines.append(' '.join(map(str, [i, *m['board'], len(c), *c, int(m['next']['bonus']), *m['counts'], *m['cycle']])))
            meta.append((m['seed'], m['move'], max(m['board']), m['board'].count(0), m['chosen']))
        k += 1
(OUT / 'positions.txt').write_text('\n'.join(lines) + '\n')
A = np.array(meta, np.int64); np.savez(OUT / 'meta.npz', **{c: A[:, j] for j, c in enumerate(('seed', 'move', 'top', 'empty', 'chosen'))})
print(N, 'moves;', len(lines), 'positions from', len(set(A[:, 0])), 'games')
