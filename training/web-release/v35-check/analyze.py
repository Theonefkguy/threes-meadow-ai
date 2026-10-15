"""V35 web release check: shipped weights, cycle rule, cycle-aware search (2048 games)."""
import json, math
from pathlib import Path
D = Path('training/web-release/v35-check'); Z = 1.959963984540054
rows = {}
for f in (D / 'games').glob('c-*.jsonl'):
    for l in f.read_text().splitlines():
        r = json.loads(l); rows[r['seed']] = r
rows = [rows[s] for s in range(31300001, 31302049)]; n = len(rows)
def wilson(k, n):
    p = k / n; d = 1 + Z * Z / n; c = p + Z * Z / (2 * n); h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)); return [(c - h) / d, (c + h) / d]
def two(k1, n1, k2, n2):
    p = (k1 + k2) / (n1 + n2); se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)); z = (k1 / n1 - k2 / n2) / se
    return {'differencePP': 100 * (k1 / n1 - k2 / n2), 'p': math.erfc(abs(z) / math.sqrt(2))}
A = json.loads(Path('training/v35-cycle/results.json').read_text())['arms']['A']
old = [json.loads(l) for f in Path('training/web-release/v31-check/games').glob('q-*.jsonl') for l in f.read_text().splitlines()]
res = {'games': n}
for k in ('r3072', 'r6144', 'r12288'):
    c = sum(r[k] for r in rows)
    res[k] = {'count': c, 'rate': c / n, 'wilson95': wilson(c, n),
              'vsV35armA': two(c, n, A[k]['count'], 4096),
              'vsOldRuleV31': two(c, n, sum(r[k] for r in old), len(old))}
res['P12288given6144'] = res['r12288']['count'] / max(1, res['r6144']['count'])
res['meanScore'] = sum(r['score'] for r in rows) / n; res['meanMoves'] = sum(r['moves'] for r in rows) / n
res['bonusesPerMove'] = sum(r['bonuses'] for r in rows) / sum(r['moves'] for r in rows)
res['sanityPass'] = res['r6144']['vsV35armA']['p'] >= 0.05 and res['r12288']['vsV35armA']['p'] >= 0.05
(D / 'results.json').write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))
