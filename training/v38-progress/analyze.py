"""V38 analysis -> results.json: paired by pool state; primary E1 B vs W31, E2 P vs C (12288, exact McNemar, Holm)."""
import json, math
from pathlib import Path
L = Path('training/v38-progress'); Z = 1.959963984540054
def load(a):
    d = {}
    for f in (L / 'eval').glob(f'{a}-*.jsonl'):
        for l in f.read_text().splitlines():
            r = json.loads(l); d[r['index']] = r
    return [d[i] for i in range(2048)]
A = {a: load(a) for a in ('W31', 'B', 'C', 'P')}; n = 2048
def wilson(k):
    p = k / n; d = 1 + Z * Z / n; c = p + Z * Z / (2 * n); h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)); return [(c - h) / d, (c + h) / d]
def mcn(x, y):
    xo = sum(a['success'] and not b['success'] for a, b in zip(A[x], A[y])); yo = sum(b['success'] and not a['success'] for a, b in zip(A[x], A[y])); t = xo + yo
    p = 1.0 if t == 0 else min(1.0, 2 * sum(math.comb(t, i) for i in range(min(xo, yo) + 1)) / 2 ** t)
    d = [a['scoreGained'] - b['scoreGained'] for a, b in zip(A[x], A[y])]; m = sum(d) / n; se = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1)) / math.sqrt(n)
    return {'onlyFirst': xo, 'onlySecond': yo, 'differencePP': 100 * (xo - yo) / n, 'p': p, 'scoreDifference': m, 'scoreApprox95': [m - Z * se, m + Z * se]}
res = {'arms': {a: {'12288': sum(r['success'] for r in A[a]), 'rate': sum(r['success'] for r in A[a]) / n, 'wilson95': wilson(sum(r['success'] for r in A[a])),
                    'meanScoreGained': sum(r['scoreGained'] for r in A[a]) / n, 'meanMoves': sum(r['moves'] for r in A[a]) / n,
                    'cpuMsPerMove': 1000 * sum(r['searchCPU'] for r in A[a]) / sum(r['moves'] for r in A[a])} for a in A}}
prim = {'E1 B vs W31': mcn('B', 'W31'), 'E2 P vs C': mcn('P', 'C')}
order = sorted(prim, key=lambda k: prim[k]['p']); run = 0
for i, k in enumerate(order): run = max(run, min(1.0, (2 - i) * prim[k]['p'])); prim[k]['holmP'] = run
res['primary'] = prim; res['secondary'] = {'P vs W31': mcn('P', 'W31'), 'C vs W31': mcn('C', 'W31'), 'P vs B': mcn('P', 'B')}
(L / 'results.json').write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))
