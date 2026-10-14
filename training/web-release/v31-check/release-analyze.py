"""Check 4 analysis: V23 vs quantized V31 on seeds 31300001-31302048. Writes release-results.json."""
from pathlib import Path
import json, math
HERE = Path(__file__).resolve().parent; Z = 1.959963984540054
def load(pat):
    out = {}
    for f in (HERE / 'games').glob(pat):
        for l in f.read_text().splitlines():
            r = json.loads(l); out[r['seed']] = r
    return [out[s] for s in range(31300001, 31302049)]
a, b = load('q-*.jsonl'), load('v23-*.jsonl'); n = len(a)
def wilson(k):
    p = k / n; d = 1 + Z * Z / n; c = p + Z * Z / (2 * n); h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)); return [(c - h) / d, (c + h) / d]
def mcn(m):
    ao = sum(x[m] and not y[m] for x, y in zip(a, b)); bo = sum(y[m] and not x[m] for x, y in zip(a, b)); t = ao + bo
    return {'V31only': ao, 'V23only': bo, 'differencePP': 100 * (ao - bo) / n, 'p': 1.0 if t == 0 else min(1.0, 2 * sum(math.comb(t, i) for i in range(min(ao, bo) + 1)) / 2 ** t)}
d = [x['score'] - y['score'] for x, y in zip(a, b)]; m = sum(d) / n; se = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1)) / math.sqrt(n)
prim = {'P1 reach 6144': mcn('r6144'), 'P2 reach 12288': mcn('r12288'), 'P3 final score': {'meanDifference': m, 'approx95': [m - Z * se, m + Z * se], 'p': math.erfc(abs(m / se) / math.sqrt(2))}}
order = sorted(prim, key=lambda k: prim[k]['p']); run = 0
for i, k in enumerate(order):
    run = max(run, min(1.0, (len(order) - i) * prim[k]['p'])); prim[k]['holmP'] = run
arms = {name: {**{k: {'count': sum(r[k] for r in rows), 'rate': sum(r[k] for r in rows) / n, 'wilson95': wilson(sum(r[k] for r in rows))} for k in ('r1536', 'r3072', 'r6144', 'r12288')},
               'meanScore': sum(r['score'] for r in rows) / n, 'cpuMsPerMove': 1000 * sum(r['searchCPU'] for r in rows) / sum(r['moves'] for r in rows)} for name, rows in (('V31q', a), ('V23', b))}
res = {'games': n, 'arms': arms, 'primary': prim, 'secondary r3072': mcn('r3072')}
(HERE / 'release-results.json').write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))
