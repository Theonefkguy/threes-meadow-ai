"""V29 analysis (repository root): python3 training/v29-five/analyze.py [curve|confirm|all]
Fixed before any outcome (analysis-pre-results-manifest.json). Writes results.json.
"""
from pathlib import Path
import json, math, sys

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / 'protocol.json').read_text())
Z = 1.959963984540054

def wilson(k, n):
    p = k / n; d = 1 + Z * Z / n; c = p + Z * Z / (2 * n); h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return [(c - h) / d, (c + h) / d]

def mcnemar(a, b):
    ao = sum(x and not y for x, y in zip(a, b)); bo = sum(y and not x for x, y in zip(a, b)); n = ao + bo
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(ao, bo) + 1)) / 2 ** n)
    return ao, bo, p

def paired(x, y):
    d = [a - b for a, b in zip(x, y)]; n = len(d); m = sum(d) / n
    se = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1)) / math.sqrt(n)
    z = m / se if se else 0.0
    return {'meanDifference': m, 'approx95': [m - Z * se, m + Z * se], 'p': math.erfc(abs(z) / math.sqrt(2))}

def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0.0] * len(ps); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); out[i] = run
    return out

def load(pool, tag):
    n = P['pools'][pool]['size']; seeds = [int(l.split()[0]) for l in (HERE / 'pools' / f'{pool}3072.txt').read_text().splitlines()]
    rows = {}
    for f in (HERE / 'eval').glob(f'{pool}-{tag}-*.jsonl'):
        for l in f.read_text().splitlines():
            r = json.loads(l); rows[r['index']] = r
    rows = [rows[i] for i in range(n)]
    assert [r['sourceSeed'] for r in rows] == seeds and all(r['startRank'] == 13 and r['depth'] == 5 for r in rows)
    assert all(r['success'] == (r['finalRank'] >= 14) and (r['success'] or r['died']) for r in rows)
    return rows

def stats(rows):
    n = len(rows); k = sum(r['success'] for r in rows)
    return {'n': n, 'reach6144': k, 'rate': k / n, 'wilson95': wilson(k, n), 'meanScoreGained': sum(r['scoreGained'] for r in rows) / n,
            'meanMoves': sum(r['moves'] for r in rows) / n, 'avoidableCertainDeath': sum(r['lastKind'] == 2 for r in rows),
            'cpuMsPerMove': 1000 * sum(r['searchCPU'] for r in rows) / sum(r['moves'] for r in rows)}

def compare(x, y):
    ao, bo, p = mcnemar([r['success'] for r in x], [r['success'] for r in y])
    return {'firstOnly': ao, 'secondOnly': bo, 'differencePP': 100 * (ao - bo) / len(x), 'mcnemarExactP': p,
            'scoreGained': paired([r['scoreGained'] for r in x], [r['scoreGained'] for r in y])}

def curve():
    tags = ['CUR'] + [f'{a}-{c}' for a in P['arms'] for c in [*map(str, P['training']['checkpoints']), 'final']]
    g = {t: load('curve', t) for t in tags}
    return {'description': 'descriptive only', 'arms': {t: stats(g[t]) for t in tags},
            'vsCUR': {t: compare(g[t], g['CUR']) for t in tags if t != 'CUR'},
            'RvsC': {c: compare(g[f'R-{c}'], g[f'C-{c}']) for c in [*map(str, P['training']['checkpoints']), 'final']}}

def confirm():
    g = {t: load('confirm', t) for t in ('CUR', 'C-final', 'R-final')}
    p1 = compare(g['R-final'], g['C-final'])
    sec = {'P2 R-final vs CUR': compare(g['R-final'], g['CUR']), 'P3 C-final vs CUR': compare(g['C-final'], g['CUR'])}
    for k, h in zip(sec, holm([v['mcnemarExactP'] for v in sec.values()])):
        sec[k]['holmP'] = h
    p3 = sec['P3 C-final vs CUR']
    return {'arms': {t: stats(g[t]) for t in g},
            'P1 R-final vs C-final': {**p1, 'significant': p1['mcnemarExactP'] < 0.05,
                                      'favours': 'R' if p1['firstOnly'] > p1['secondOnly'] else 'C' if p1['secondOnly'] > p1['firstOnly'] else 'tie'},
            'secondary': sec,
            'C adoption native gates': {'holmSignificantGain': p3['holmP'] < 0.05 and p3['firstOnly'] > p3['secondOnly'],
                                        'scoreNotSignificantlyWorse': not (p3['scoreGained']['p'] < 0.05 and p3['scoreGained']['meanDifference'] < 0)}}

if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    path = HERE / 'results.json'; res = json.loads(path.read_text()) if path.exists() else {}
    if what in ('curve', 'all'): res['curve'] = curve()
    if what in ('confirm', 'all'): res['confirm'] = confirm()
    path.write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1)[:4000])
