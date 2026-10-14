"""Check 1 analysis: quantized vs float V31 on the V31 whole-game seeds. Writes results.json."""
from pathlib import Path
import json, math
HERE = Path(__file__).resolve().parent
FLOAT = Path('/Users/luipy/Projects/active/threes-meadow-ai/training/v31-stages/eval')
def load(d, pat):
    out = {}
    for f in d.glob(pat):
        for l in f.read_text().splitlines():
            r = json.loads(l); out[r['seed']] = r
    return [out[s] for s in range(31300001, 31302049)]
q, f = load(HERE / 'games', 'q-*.jsonl'), load(FLOAT, 'games-W31-*.jsonl')
def mcn(a, b):
    ao = sum(x and not y for x, y in zip(a, b)); bo = sum(y and not x for x, y in zip(a, b)); n = ao + bo
    return {'quantOnly': ao, 'floatOnly': bo, 'p': 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(ao, bo) + 1)) / 2 ** n)}
d = [a['score'] - b['score'] for a, b in zip(q, f)]; m = sum(d) / len(d)
se = math.sqrt(sum((v - m) ** 2 for v in d) / (len(d) - 1)) / math.sqrt(len(d)); z = m / se if se else 0
keys = [k for k in q[0] if 'CPU' not in k]
res = {'games': len(q), 'identicalGames': sum(all(a[k] == b[k] for k in keys) for a, b in zip(q, f)),
       'rates': {name: {m_: sum(r[m_] for r in rows) for m_ in ('r3072', 'r6144', 'r12288')} for name, rows in (('quantized', q), ('float', f))},
       'r6144': mcn([r['r6144'] for r in q], [r['r6144'] for r in f]), 'r12288': mcn([r['r12288'] for r in q], [r['r12288'] for r in f]),
       'score': {'meanDifference': m, 'approx95': [m - 1.96 * se, m + 1.96 * se], 'p': math.erfc(abs(z) / math.sqrt(2)) if se else 1.0}}
worse = lambda t: t['p'] < 0.05 and t['quantOnly'] < t['floatOnly']
res['pass'] = not (worse(res['r6144']) or worse(res['r12288']) or (res['score']['p'] < 0.05 and m < 0))
(HERE / 'results.json').write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))
