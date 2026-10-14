"""V34 analysis: each arm vs the shipped float16 model (check-1 games) on seeds 31300001-31302048."""
import json, math
from pathlib import Path
L = Path('training/v34-lossy'); P = json.loads((L / 'protocol.json').read_text()); SEEDS = range(31300001, 31302049)
def load(d, pat):
    out = {}
    for f in Path(d).glob(pat):
        for l in f.read_text().splitlines():
            r = json.loads(l); out[r['seed']] = r
    return [out[s] for s in SEEDS]
base = load('training/web-release/v31-check/games', 'q-*.jsonl'); n = len(base); res = {'reference': 'shipped V31 (float32 base, float16 residuals)', 'arms': {}}
def mcn(a, b, m):
    ao = sum(x[m] and not y[m] for x, y in zip(a, b)); bo = sum(y[m] and not x[m] for x, y in zip(a, b)); t = ao + bo
    return {'armOnly': ao, 'shippedOnly': bo, 'differencePP': 100 * (ao - bo) / n, 'p': 1.0 if t == 0 else min(1.0, 2 * sum(math.comb(t, i) for i in range(min(ao, bo) + 1)) / 2 ** t)}
for arm in P['arms']:
    try: g = load(L / 'games', f'{arm}-*.jsonl')
    except KeyError: continue
    d = [a['score'] - b['score'] for a, b in zip(g, base)]; m = sum(d) / n; se = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1)) / math.sqrt(n)
    keys = [k for k in g[0] if 'CPU' not in k]
    r = {'identicalGames': sum(all(a[k] == b[k] for k in keys) for a, b in zip(g, base)), 'r6144': mcn(g, base, 'r6144'), 'r12288': mcn(g, base, 'r12288'),
         'score': {'meanDifference': m, 'approx95': [m - 1.96 * se, m + 1.96 * se], 'p': math.erfc(abs(m / se) / math.sqrt(2)) if se else 1.0},
         'counts': {k: sum(x[k] for x in g) for k in ('r3072', 'r6144', 'r12288')}, 'downloadMB': P['arms'][arm]['downloadMB']}
    worse = lambda t: t['p'] < 0.05 and t['armOnly'] < t['shippedOnly']
    r['pass'] = not (worse(r['r6144']) or worse(r['r12288']) or (r['score']['p'] < 0.05 and m < 0)) and r['r6144']['differencePP'] >= -1.0
    res['arms'][arm] = r
res['shippedCounts'] = {k: sum(x[k] for x in base) for k in ('r3072', 'r6144', 'r12288')}
passing = [a for a in res['arms'] if res['arms'][a]['pass']]
res['choice'] = min(passing, key=lambda a: res['arms'][a]['downloadMB']) if passing else None
(L / 'results.json').write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))
