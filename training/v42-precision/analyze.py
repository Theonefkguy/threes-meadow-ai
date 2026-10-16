"""V42: full precision (float32 residuals) vs the website's float16 residuals, same seeds and search
(V41 arm D). Paired by seed -> results.json."""
import json, math
from pathlib import Path
L = Path('training/v42-precision'); Z = 1.959963984540054
def load(pat):
    d = {r['seed']: r for f in Path().glob(pat) for r in map(json.loads, f.read_text().splitlines())}
    return [d[s] for s in range(41100001, 41104097)]
F, H = load(f'{L}/games/F-*.jsonl'), load('training/v41-stage3-depth/whole/D-*.jsonl'); n = len(F)
def wilson(k):
    p = k / n; d = 1 + Z * Z / n; c = p + Z * Z / (2 * n); h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)); return [(c - h) / d, (c + h) / d]
def mcn(k):
    a = sum(f[k] and not h[k] for f, h in zip(F, H)); b = sum(h[k] and not f[k] for f, h in zip(F, H)); t = a + b
    return {'onlyFull': a, 'onlyHalf': b, 'differencePP': 100 * (a - b) / n, 'p': 1.0 if t == 0 else min(1.0, 2 * sum(math.comb(t, i) for i in range(min(a, b) + 1)) / 2 ** t)}
def arm(rows):
    six = [r for r in rows if r['r6144']]
    return {**{k: {'count': sum(r[k] for r in rows), 'rate': sum(r[k] for r in rows) / n, 'wilson95': wilson(sum(r[k] for r in rows))} for k in ('r3072', 'r6144', 'r12288')},
            'P12288given6144': sum(r['r12288'] for r in six) / len(six), 'meanScore': sum(r['score'] for r in rows) / n}
d = [f['score'] - h['score'] for f, h in zip(F, H)]; m = sum(d) / n; se = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1)) / math.sqrt(n)
res = {'games': n, 'full': arm(F), 'half': arm(H), 'primary 12288': mcn('r12288'), 'secondary': {'r6144': mcn('r6144'), 'r3072': mcn('r3072'),
       'score full - half': {'mean': m, 'approx95': [m - Z * se, m + Z * se], 'p': math.erfc(abs(m / se) / math.sqrt(2))}},
       'identicalGames': sum(f['moves'] == h['moves'] and f['score'] == h['score'] for f, h in zip(F, H))}
(L / 'results.json').write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))
