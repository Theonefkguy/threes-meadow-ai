"""V66 analysis (setup in RL-V66-DIRECT-DEPTH.md): per start depth of the iterations (1 = current website search, 6 =
depth 6 only), share of positions with the same move and bit-identical root values as start 1, largest relative
value difference, and CPU time / nodes relative to start 1, overall and by empty cells.  -> results.json"""
import json
from pathlib import Path
import numpy as np
L = Path(__file__).resolve().parent
A = np.loadtxt(L / 'direct.txt', dtype=np.float64); meta = json.load(open(L.parent / 'v62-time-share/positions.json'))
assert len(A) == 6000 and (A[:, 0] == np.arange(len(A))).all()
starts = [1, 3, 4, 5, 6]; col = lambda j, k: A[:, 1 + 7 * j + k]
ms = np.stack([col(j, 0) for j in range(5)], 1); nodes = np.stack([col(j, 1) for j in range(5)], 1)
dirs = np.stack([col(j, 2) for j in range(5)], 1).astype(int); V = np.stack([np.stack([col(j, 3 + q) for q in range(4)], 1) for j in range(5)], 1)
empty = np.array([m['empty'] for m in meta])
legal = V[:, 0] > -1e299
out = {'positions': len(A), 'ms1': float(ms[:, 0].mean()), 'arms': {}}
for j, s in enumerate(starts):
    same = (V[:, j] == V[:, 0]).all(1)
    rel = np.where(legal, np.abs(V[:, j] - V[:, 0]) / np.maximum(np.abs(V[:, 0]), 1), 0).max(1)
    r = {'sameMove': float((dirs[:, j] == dirs[:, 0]).mean()), 'moveDiffers': int((dirs[:, j] != dirs[:, 0]).sum()),
         'identicalValues': float(same.mean()), 'maxRelDiff': float(rel.max()), 'medianRelDiffWhenDiffer': float(np.median(rel[~same])) if (~same).any() else 0.0,
         'msShare': float(ms[:, j].sum() / ms[:, 0].sum()), 'nodesShare': float(nodes[:, j].sum() / nodes[:, 0].sum()), 'byEmpty': {}}
    for lo, hi in [(0, 0), (1, 2), (3, 4), (5, 16)]:
        k = (empty >= lo) & (empty <= hi)
        r['byEmpty']['%d-%d' % (lo, hi)] = {'share': float(k.mean()), 'msShare': float(ms[k, j].sum() / ms[k, 0].sum())}
    out['arms'][s] = r
(L / 'results.json').write_text(json.dumps(out, indent=1) + '\n')
for s, r in out['arms'].items():
    print('start %d: same move %.4f (%d differ), identical values %.4f, max rel %.2e, time %.3f, nodes %.3f' % (
        s, r['sameMove'], r['moveDiffers'], r['identicalValues'], r['maxRelDiff'], r['msShare'], r['nodesShare']),
        ' '.join('%s:%.3f' % (k, v['msShare']) for k, v in r['byEmpty'].items()))
