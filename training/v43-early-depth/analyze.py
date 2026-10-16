"""V43: 6-ply / 0.005 from the first move (E) vs the website setting (V41 arm D: 5-ply / 0.01 until a 6144,
then 6-ply / 0.005), same seeds and weights. Paired by seed -> results.json."""
import json, math
from pathlib import Path
L = Path('training/v43-early-depth'); Z = 1.959963984540054
def load(pat):
    d = {r['seed']: r for f in Path().glob(pat) for r in map(json.loads, f.read_text().splitlines())}
    return [d[s] for s in range(41100001, 41104097)]
E, D = load(f'{L}/games/E-*.jsonl'), load('training/v41-stage3-depth/whole/D-*.jsonl'); n = len(E)
def wilson(k):
    p = k / n; d = 1 + Z * Z / n; c = p + Z * Z / (2 * n); h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)); return [(c - h) / d, (c + h) / d]
def mcn(k):
    a = sum(e[k] and not d[k] for e, d in zip(E, D)); b = sum(d[k] and not e[k] for e, d in zip(E, D)); t = a + b
    return {'onlyE': a, 'onlyD': b, 'differencePP': 100 * (a - b) / n, 'p': 1.0 if t == 0 else min(1.0, 2 * sum(math.comb(t, i) for i in range(min(a, b) + 1)) / 2 ** t)}
def arm(rows):
    six = [r for r in rows if r['r6144']]
    out = {k: {'count': sum(r[k] for r in rows), 'rate': sum(r[k] for r in rows) / n, 'wilson95': wilson(sum(r[k] for r in rows))} for k in ('r1536', 'r3072', 'r6144', 'r12288')}
    out.update(P12288given6144=sum(r['r12288'] for r in six) / len(six), meanScore=sum(r['score'] for r in rows) / n,
               nativeMsPerMove=1000 * sum(r['searchCPU'] for r in rows) / sum(r['moves'] for r in rows))
    return out
prim = {'P1 reach 6144': mcn('r6144'), 'P2 reach 12288': mcn('r12288')}
order = sorted(prim, key=lambda k: prim[k]['p']); run = 0
for i, k in enumerate(order): run = max(run, min(1.0, (2 - i) * prim[k]['p'])); prim[k]['holmP'] = run
d = [e['score'] - x['score'] for e, x in zip(E, D)]; m = sum(d) / n; se = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1)) / math.sqrt(n)
res = {'games': n, 'E': arm(E), 'D': arm(D), 'primary': prim, 'secondary': {'r3072': mcn('r3072'), 'r1536': mcn('r1536'),
       'score E - D': {'mean': m, 'approx95': [m - Z * se, m + Z * se], 'p': math.erfc(abs(m / se) / math.sqrt(2))}}}
(L / 'results.json').write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))
