"""V41 step 2 analysis -> results.json and the web setting choice (rule in protocol.json)."""
import json, math
from pathlib import Path
L = Path('training/v41-stage3-depth'); P = json.loads((L / 'protocol.json').read_text()); Z = 1.959963984540054
def load(pat):
    d = {r['index']: r for f in Path().glob(pat) for r in map(json.loads, f.read_text().splitlines())}
    return [d[i] for i in range(2048)]
A = {'base': load('training/v39-endgame/games/d-*.jsonl'), 'S': load('training/v40-late-depth/games/S-*.jsonl'),
     'D6': load(f'{L}/games/D6-*.jsonl'), 'T5': load(f'{L}/games/T5-*.jsonl')}; n = 2048
def mcn(x, y):
    xo = sum(a['success'] and not b['success'] for a, b in zip(A[x], A[y])); yo = sum(b['success'] and not a['success'] for a, b in zip(A[x], A[y])); t = xo + yo
    p = 1.0 if t == 0 else min(1.0, 2 * sum(math.comb(t, i) for i in range(min(xo, yo) + 1)) / 2 ** t)
    d = [a['scoreGained'] - b['scoreGained'] for a, b in zip(A[x], A[y])]; m = sum(d) / n; se = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1)) / math.sqrt(n)
    return {'onlyFirst': xo, 'onlySecond': yo, 'differencePP': 100 * (xo - yo) / n, 'p': p, 'scoreDifference': m, 'scoreApprox95': [m - Z * se, m + Z * se]}
arms = {}
for a, rows in A.items():
    k = sum(r['success'] for r in rows); arms[a] = {'12288': k, 'rate': k / n, 'meanScoreGained': sum(r['scoreGained'] for r in rows) / n}
    if a != 'base': arms[a]['nativeMsPerMove'] = 1000 * sum(r['searchCPU'] for r in rows) / sum(r['moves'] for r in rows)
prim = {'D6 vs base': mcn('D6', 'base'), 'T5 vs base': mcn('T5', 'base')}
order = sorted(prim, key=lambda k: prim[k]['p']); run = 0
for i, k in enumerate(order): run = max(run, min(1.0, (2 - i) * prim[k]['p'])); prim[k]['holmP'] = run
vsS = {a: mcn(a, 'S') for a in ('D6', 'T5')}
browser = {(r['depth'], r['thr']): r for r in json.loads((L / 'browser-6144.json').read_text())['results']}
setting = {'S': (6, 0.005), 'D6': (6, 0.01), 'T5': (5, 0.005)}
def ok(a):
    b = browser[setting[a]]
    if b['p95Ms'] > 160 or b['fullDepth'] < 0.99 * b['n']: return False
    if a == 'S': return True
    return prim[f'{a} vs base']['holmP'] < 0.05 and prim[f'{a} vs base']['differencePP'] > 0 and not (vsS[a]['p'] < 0.05 and vsS[a]['differencePP'] < 0)
eligible = [a for a in setting if ok(a)]
choice = min(eligible, key=lambda a: browser[setting[a]]['meanMs']) if eligible else None
res = {'arms': arms, 'primary': prim, 'vsS': vsS, 'eligible': eligible, 'choice': choice, 'choiceSetting': setting.get(choice), 'rule': P['choice']}
(L / 'results.json').write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))
