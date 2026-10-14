"""V35 analysis: cycle-aware (A) vs unaware (U) search on the same seeds under the cycle rule."""
import json, math
from pathlib import Path
L = Path('training/v35-cycle'); P = json.loads((L / 'protocol.json').read_text()); first, last = P['seeds']; Z = 1.959963984540054
def load(arm):
    out = {}
    for f in (L / 'games').glob(f'{arm}-*.jsonl'):
        for l in f.read_text().splitlines():
            r = json.loads(l); out[r['seed']] = r
    return [out[s] for s in range(first, last + 1)]
u, a = load('U'), load('A'); n = len(u)
def wilson(k):
    p = k / n; d = 1 + Z * Z / n; c = p + Z * Z / (2 * n); h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)); return [(c - h) / d, (c + h) / d]
def mcn(m):
    ao = sum(x[m] and not y[m] for x, y in zip(a, u)); bo = sum(y[m] and not x[m] for x, y in zip(a, u)); t = ao + bo
    return {'Aonly': ao, 'Uonly': bo, 'differencePP': 100 * (ao - bo) / n, 'p': 1.0 if t == 0 else min(1.0, 2 * sum(math.comb(t, i) for i in range(min(ao, bo) + 1)) / 2 ** t)}
d = [x['score'] - y['score'] for x, y in zip(a, u)]; m = sum(d) / n; se = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1)) / math.sqrt(n)
prim = {'P1 reach 6144': mcn('r6144'), 'P2 reach 12288': mcn('r12288'), 'P3 final score': {'meanDifference': m, 'approx95': [m - Z * se, m + Z * se], 'p': math.erfc(abs(m / se) / math.sqrt(2))}}
order = sorted(prim, key=lambda k: prim[k]['p']); run = 0
for i, k in enumerate(order):
    run = max(run, min(1.0, (len(order) - i) * prim[k]['p'])); prim[k]['holmP'] = run
arms = {name: {**{k: {'count': sum(r[k] for r in rows), 'rate': sum(r[k] for r in rows) / n, 'wilson95': wilson(sum(r[k] for r in rows))} for k in ('r3072', 'r6144', 'r12288')},
               'P12288given6144': sum(r['r12288'] for r in rows) / max(1, sum(r['r6144'] for r in rows)),
               'meanScore': sum(r['score'] for r in rows) / n, 'meanMoves': sum(r['moves'] for r in rows) / n,
               'cpuMsPerMove': 1000 * sum(r['searchCPU'] for r in rows) / sum(r['moves'] for r in rows)} for name, rows in (('A', a), ('U', u))}
res = {'games': n, 'arms': arms, 'primary': prim, 'secondary r3072': mcn('r3072'),
       'identicalGames': sum(x['moves'] == y['moves'] and x['score'] == y['score'] for x, y in zip(a, u))}
(L / 'results.json').write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))
