"""V40 analysis -> results.json (paired by pool state vs the V39 baseline)."""
import json, math
from pathlib import Path
L = Path('training/v40-late-depth'); Z = 1.959963984540054
def load(pat):
    d = {r['index']: r for f in Path().glob(pat) for r in map(json.loads, f.read_text().splitlines())}
    return [d[i] for i in range(2048)]
A = {'base': load('training/v39-endgame/games/d-*.jsonl'), 'L': load(f'{L}/games/L-*.jsonl'), 'S': load(f'{L}/games/S-*.jsonl')}; n = 2048
def wilson(k):
    p = k / n; d = 1 + Z * Z / n; c = p + Z * Z / (2 * n); h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)); return [(c - h) / d, (c + h) / d]
def mcn(x, y='base'):
    xo = sum(a['success'] and not b['success'] for a, b in zip(A[x], A[y])); yo = sum(b['success'] and not a['success'] for a, b in zip(A[x], A[y])); t = xo + yo
    p = 1.0 if t == 0 else min(1.0, 2 * sum(math.comb(t, i) for i in range(min(xo, yo) + 1)) / 2 ** t)
    d = [a['scoreGained'] - b['scoreGained'] for a, b in zip(A[x], A[y])]; m = sum(d) / n; se = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1)) / math.sqrt(n)
    return {'onlyArm': xo, 'onlyBase': yo, 'differencePP': 100 * (xo - yo) / n, 'p': p, 'scoreDifference': m, 'scoreApprox95': [m - Z * se, m + Z * se]}
def arm(a):
    rows = A[a]; k = sum(r['success'] for r in rows); m768 = [r for r in rows if r['m768'] >= 0]
    out = {'12288': k, 'rate': k / n, 'wilson95': wilson(k), 'meanScoreGained': sum(r['scoreGained'] for r in rows) / n,
           'reachM768': len(m768), 'P12288givenM768': sum(r['success'] for r in m768) / len(m768), 'reachReady': sum(r['ready'] >= 0 for r in rows)}
    if a != 'base':
        dm = sum(r['deepMoves'] for r in rows); out['deepMoves'] = dm; out['deepMsPerMove'] = 1000 * sum(r['deepCPU'] for r in rows) / dm if dm else None
        out['shallowMsPerMove'] = 1000 * (sum(r['searchCPU'] for r in rows) - sum(r['deepCPU'] for r in rows)) / max(1, sum(r['moves'] for r in rows) - dm)
    return out
res = {'arms': {a: arm(a) for a in A}, 'primary': {'L vs base': mcn('L'), 'S vs base': mcn('S')}}
order = sorted(res['primary'], key=lambda k: res['primary'][k]['p']); run = 0
for i, k in enumerate(order): run = max(run, min(1.0, (2 - i) * res['primary'][k]['p'])); res['primary'][k]['holmP'] = run
res['secondary'] = {'S vs L': mcn('S', 'L')}
(L / 'results.json').write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))
