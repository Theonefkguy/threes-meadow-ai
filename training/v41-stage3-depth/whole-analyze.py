"""V41 step 3 analysis -> whole-results.json (paired by seed, D vs W)."""
import json, math
from pathlib import Path
L = Path('training/v41-stage3-depth'); Z = 1.959963984540054
def load(a):
    d = {r['seed']: r for f in (L / 'whole').glob(f'{a}-*.jsonl') for r in map(json.loads, f.read_text().splitlines())}
    return [d[s] for s in range(41100001, 41104097)]
W, D = load('W'), load('D'); n = len(W)
assert all(w['r6144'] == d['r6144'] and w['turn6144'] == d['turn6144'] for w, d in zip(W, D)), 'games differ before the first 6144'
def wilson(k):
    p = k / n; d = 1 + Z * Z / n; c = p + Z * Z / (2 * n); h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)); return [(c - h) / d, (c + h) / d]
def arm(rows):
    out = {k: {'count': sum(r[k] for r in rows), 'rate': sum(r[k] for r in rows) / n, 'wilson95': wilson(sum(r[k] for r in rows))} for k in ('r1536', 'r3072', 'r6144', 'r12288')}
    for k, name in (('c2', '6144+3072'), ('c3', '6144+3072+1536'), ('c4', '6144+3072+1536+768')):
        c = sum(r[k] >= 0 for r in rows); out[name] = {'count': c, 'rate': c / n}
    six = [r for r in rows if r['r6144']]; out['P12288given6144'] = sum(r['r12288'] for r in six) / len(six)
    out['meanScore'] = sum(r['score'] for r in rows) / n; out['meanMoves'] = sum(r['moves'] for r in rows) / n
    out['nativeMsPerLateMove'] = 1000 * sum(r['lateCPU'] for r in rows) / max(1, sum(r['lateMoves'] for r in rows))
    return out
xo = sum(d['r12288'] and not w['r12288'] for w, d in zip(W, D)); yo = sum(w['r12288'] and not d['r12288'] for w, d in zip(W, D)); t = xo + yo
p = min(1.0, 2 * sum(math.comb(t, i) for i in range(min(xo, yo) + 1)) / 2 ** t)
diff = [d['score'] - w['score'] for w, d in zip(W, D)]; m = sum(diff) / n; se = math.sqrt(sum((v - m) ** 2 for v in diff) / (n - 1)) / math.sqrt(n)
res = {'games': n, 'W': arm(W), 'D': arm(D), 'primary 12288 D vs W': {'onlyD': xo, 'onlyW': yo, 'differencePP': 100 * (xo - yo) / n, 'p': p},
       'score D - W': {'mean': m, 'approx95': [m - Z * se, m + Z * se], 'p': math.erfc(abs(m / se) / math.sqrt(2))}}
(L / 'whole-results.json').write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))
