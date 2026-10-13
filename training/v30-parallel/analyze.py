"""V30 analysis (repository root): python3 training/v30-parallel/analyze.py select | games
Fixed before any outcome (analysis-pre-results-manifest.json). Standard library only.
"""
from pathlib import Path
import json, math, sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V29 = ROOT / 'training/v29-five'
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

def rows(directory, pattern, key, expect):
    out = {}
    for f in directory.glob(pattern):
        for l in f.read_text().splitlines():
            r = json.loads(l); out[r[key]] = r
    return [out[k] for k in expect]

def select():
    n = P['select']['size']; seeds = [int(l.split()[0]) for l in (V29 / 'pools/curve3072.txt').read_text().splitlines()]
    def load(tag):
        if tag in ('CUR', 'V29R'):
            r = rows(V29 / 'eval', f"curve-{'CUR' if tag == 'CUR' else 'R-final'}-*.jsonl", 'index', range(n))
        else:
            r = rows(HERE / 'eval', f'select-{tag}-*.jsonl', 'index', range(n))
        assert [x['sourceSeed'] for x in r] == seeds and all(x['startRank'] == 13 and x['depth'] == 5 for x in r)
        return r
    tags = ['CUR', *P['candidates']]; g = {t: load(t) for t in tags}
    count = {t: sum(r['success'] for r in g[t]) for t in tags}
    order = P['candidates']  # ascending training; ties go to the later (more trained) one
    winner = max(order, key=lambda t: (count[t], order.index(t)))
    res = {'winner': winner, 'rule': P['select']['rule'],
           'arms': {t: {'reach6144': count[t], 'rate': count[t] / n, 'wilson95': wilson(count[t], n),
                        'meanScoreGained': sum(r['scoreGained'] for r in g[t]) / n,
                        'cpuMsPerMove': 1000 * sum(r['searchCPU'] for r in g[t]) / sum(r['moves'] for r in g[t])} for t in tags},
           'descriptive': {f'{t} vs V29R': dict(zip(('firstOnly', 'secondOnly', 'mcnemarExactP'), mcnemar([r['success'] for r in g[t]], [r['success'] for r in g['V29R']])))
                           for t in P['candidates'] if t != 'V29R'}}
    (HERE / 'select.json').write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))

def games():
    first, last = P['games']['seeds']; seeds = range(first, last + 1)
    g = {a: rows(HERE / 'eval', f'games-{a}-*.jsonl', 'seed', seeds) for a in ('CUR', 'W')}
    n = len(seeds); res = {'winner': json.loads((HERE / 'select.json').read_text())['winner'], 'games': n, 'arms': {}}
    for a in g:
        res['arms'][a] = {m: {'count': sum(r[m] for r in g[a]), 'rate': sum(r[m] for r in g[a]) / n, 'wilson95': wilson(sum(r[m] for r in g[a]), n)}
                          for m in ('r1536', 'r3072', 'r6144', 'r12288')}
        res['arms'][a].update({'meanScore': sum(r['score'] for r in g[a]) / n, 'meanMoves': sum(r['moves'] for r in g[a]) / n,
                               'cpuMsPerMove': 1000 * sum(r['searchCPU'] for r in g[a]) / sum(r['moves'] for r in g[a]),
                               'maxMoveCpuMs': 1000 * max(r['maxMoveCPU'] for r in g[a]),
                               'incompleteDepthMoves': sum(sum(r['depthHist'][:5]) for r in g[a])})
    prim = {}
    for key, m in (('P1 reach 6144', 'r6144'), ('P2 reach 12288', 'r12288')):
        ao, bo, p = mcnemar([r[m] for r in g['W']], [r[m] for r in g['CUR']])
        prim[key] = {'Wonly': ao, 'CURonly': bo, 'differencePP': 100 * (ao - bo) / n, 'p': p}
    sc = paired([r['score'] for r in g['W']], [r['score'] for r in g['CUR']])
    prim['P3 final score'] = {**sc}
    for k, h in zip(prim, holm([v['p'] for v in prim.values()])):
        prim[k]['holmP'] = h
    res['primary'] = prim
    for m in ('r1536', 'r3072'):
        ao, bo, p = mcnemar([r[m] for r in g['W']], [r[m] for r in g['CUR']])
        res.setdefault('secondary', {})[m] = {'Wonly': ao, 'CURonly': bo, 'p': p}
    (HERE / 'results.json').write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))

if __name__ == '__main__':
    {'select': select, 'games': games}[sys.argv[1]]()
