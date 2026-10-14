"""V32 analysis (repository root): python3 training/v32-strong/analyze.py screen | confirm
Fixed before any outcome (analysis-pre-results-manifest.json). Standard library only.
"""
from pathlib import Path
import json, math, sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
P = json.loads((HERE / 'protocol.json').read_text())
Z = 1.959963984540054

def wilson(k, n):
    p = k / n; d = 1 + Z * Z / n; c = p + Z * Z / (2 * n); h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return [(c - h) / d, (c + h) / d]

def mcnemar(a, b):
    ao = sum(x and not y for x, y in zip(a, b)); bo = sum(y and not x for x, y in zip(a, b)); n = ao + bo
    return ao, bo, (1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(ao, bo) + 1)) / 2 ** n))

def paired(x, y):
    d = [a - b for a, b in zip(x, y)]; n = len(d); m = sum(d) / n
    se = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1)) / math.sqrt(n); z = m / se if se else 0.0
    return {'meanDifference': m, 'approx95': [m - Z * se, m + Z * se], 'p': math.erfc(abs(z) / math.sqrt(2))}

def load(directory, pattern, n, seeds):
    rows = {}
    for f in directory.glob(pattern):
        for l in f.read_text().splitlines():
            r = json.loads(l); rows[r['index']] = r
    rows = [rows[i] for i in range(n)]
    assert [r['sourceSeed'] for r in rows] == seeds and all(r['startRank'] == 14 and r['depth'] == 5 for r in rows)
    assert all(r['success'] == (r['finalRank'] >= 15) and (r['success'] or r['died']) for r in rows)
    return rows

def stats(rows):
    n = len(rows); k = sum(r['success'] for r in rows)
    return {'reach12288': k, 'rate': k / n, 'wilson95': wilson(k, n), 'meanScoreGained': sum(r['scoreGained'] for r in rows) / n,
            'meanMoves': sum(r['moves'] for r in rows) / n, 'avoidableCertainDeath': sum(r['lastKind'] == 2 for r in rows),
            'cpuMsPerMove': 1000 * sum(r['searchCPU'] for r in rows) / sum(r['moves'] for r in rows)}

def compare(x, y):
    ao, bo, p = mcnemar([r['success'] for r in x], [r['success'] for r in y])
    return {'firstOnly': ao, 'secondOnly': bo, 'differencePP': 100 * (ao - bo) / len(x), 'p': p,
            'scoreGained': paired([r['scoreGained'] for r in x], [r['scoreGained'] for r in y])}

def screen():
    sp = P['screen']; n = sp['size']; seeds = [int(l.split()[0]) for l in (ROOT / sp['pool']).read_text().splitlines()]
    g = {'W31': load(ROOT / sp['w31Results'], sp['w31Pattern'], n, seeds)}
    for a in ('T3', 'T5'):
        g[a] = load(HERE / 'eval', f'screen-{a}-*.jsonl', n, seeds)
    k = {a: sum(r['success'] for r in g[a]) for a in g}; s = {a: sum(r['scoreGained'] for r in g[a]) for a in g}
    winner = 'T5' if (k['T5'], s['T5']) > (k['T3'], s['T3']) else 'T3'
    res = {'winner': winner, 'rule': sp['rule'], 'arms': {a: stats(g[a]) for a in g},
           'descriptive': {'T3 vs W31': compare(g['T3'], g['W31']), 'T5 vs W31': compare(g['T5'], g['W31']), 'T5 vs T3': compare(g['T5'], g['T3'])}}
    (HERE / 'screen.json').write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))

def confirm():
    winner = json.loads((HERE / 'screen.json').read_text())['winner']; n = P['pool']['size']
    seeds = [int(l.split()[0]) for l in (HERE / 'pools/confirm6144.txt').read_text().splitlines()]
    w, base = load(HERE / 'eval', f'confirm-{winner}-*.jsonl', n, seeds), load(HERE / 'eval', 'confirm-W31-*.jsonl', n, seeds)
    c = compare(w, base)
    res = {'winner': winner, 'arms': {winner: stats(w), 'W31': stats(base)}, 'primary': c,
           'gate': {'significantGain': c['p'] < P['confirm']['alpha'] and c['firstOnly'] > c['secondOnly'],
                    'scoreNotSignificantlyWorse': not (c['scoreGained']['p'] < 0.05 and c['scoreGained']['meanDifference'] < 0)}}
    (HERE / 'confirm.json').write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))

if __name__ == '__main__':
    {'screen': screen, 'confirm': confirm}[sys.argv[1]]()
