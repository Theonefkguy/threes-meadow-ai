"""V31 analysis (repository root): python3 training/v31-stages/analyze.py -> results.json
Fixed before any outcome (analysis-pre-results-manifest.json). Standard library only.
"""
from pathlib import Path
import json, math

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
P = json.loads((HERE / 'protocol.json').read_text())
Z = 1.959963984540054

def wilson(k, n):
    p = k / n; d = 1 + Z * Z / n; c = p + Z * Z / (2 * n); h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return [(c - h) / d, (c + h) / d]

def mcnemar(a, b):
    ao = sum(x and not y for x, y in zip(a, b)); bo = sum(y and not x for x, y in zip(a, b)); n = ao + bo
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(ao, bo) + 1)) / 2 ** n)
    return {'W31only': ao, 'W30only': bo, 'differencePP': None, 'p': p}

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

def rows(pattern, key, expect):
    out = {}
    for f in (HERE / 'eval').glob(pattern):
        for l in f.read_text().splitlines():
            r = json.loads(l); out[r[key]] = r
    return [out[k] for k in expect]

def test(x, y, n):
    t = mcnemar(x, y); t['differencePP'] = 100 * (t['W31only'] - t['W30only']) / n; return t

def main():
    res = {}
    first, last = P['games']['seeds']; n = last - first + 1
    g = {a: rows(f'games-{a}-*.jsonl', 'seed', range(first, last + 1)) for a in ('W30', 'W31')}
    res['games'] = {'n': n, 'arms': {a: {**{m: {'count': sum(r[m] for r in g[a]), 'rate': sum(r[m] for r in g[a]) / n, 'wilson95': wilson(sum(r[m] for r in g[a]), n)}
                                           for m in ('r1536', 'r3072', 'r6144', 'r12288')},
                                        'meanScore': sum(r['score'] for r in g[a]) / n, 'meanMoves': sum(r['moves'] for r in g[a]) / n,
                                        'cpuMsPerMove': 1000 * sum(r['searchCPU'] for r in g[a]) / sum(r['moves'] for r in g[a]),
                                        'maxMoveCpuMs': 1000 * max(r['maxMoveCPU'] for r in g[a]),
                                        'incompleteDepthMoves': sum(sum(r['depthHist'][:5]) for r in g[a])} for a in g}}
    prim = {'P1 reach 6144': test([r['r6144'] for r in g['W31']], [r['r6144'] for r in g['W30']], n),
            'P2 reach 12288': test([r['r12288'] for r in g['W31']], [r['r12288'] for r in g['W30']], n),
            'P3 final score': paired([r['score'] for r in g['W31']], [r['score'] for r in g['W30']])}
    for k, h in zip(prim, holm([v['p'] for v in prim.values()])):
        prim[k]['holmP'] = h
    res['games']['primary'] = prim
    res['games']['secondary'] = {m: test([r[m] for r in g['W31']], [r[m] for r in g['W30']], n) for m in ('r1536', 'r3072')}
    cond = {}
    for which, spec in P['conditional'].items():
        seeds = [int(l.split()[0]) for l in (ROOT / spec['pool']).read_text().splitlines()]
        c = {a: rows(f'{which}-{a}-*.jsonl', 'index', range(spec['size'])) for a in ('W30', 'W31')}
        for a in c:
            assert [r['sourceSeed'] for r in c[a]] == seeds and all(r['startRank'] == spec['target'] - 1 for r in c[a])
        k = {a: sum(r['success'] for r in c[a]) for a in c}
        cond[which] = {'endpoint': spec['endpoint'], 'n': spec['size'],
                       'arms': {a: {'success': k[a], 'rate': k[a] / spec['size'], 'wilson95': wilson(k[a], spec['size']),
                                    'meanScoreGained': sum(r['scoreGained'] for r in c[a]) / spec['size'],
                                    'avoidableCertainDeath': sum(r['lastKind'] == 2 for r in c[a]),
                                    'cpuMsPerMove': 1000 * sum(r['searchCPU'] for r in c[a]) / sum(r['moves'] for r in c[a])} for a in c},
                       'test': test([r['success'] for r in c['W31']], [r['success'] for r in c['W30']], spec['size']),
                       'scoreGained': paired([r['scoreGained'] for r in c['W31']], [r['scoreGained'] for r in c['W30']])}
    for k, h in zip(cond, holm([v['test']['p'] for v in cond.values()])):
        cond[k]['test']['holmP'] = h
    res['conditional'] = cond
    if 'v27Reproduction' in P['conditional']['c3']:
        v27 = {}
        for f in (ROOT / 'training/v27-stage3/eval').glob('confirm-CUR-*.jsonl'):
            for l in f.read_text().splitlines():
                r = json.loads(l); v27[r['index']] = r
        w30 = rows('c3-W30-*.jsonl', 'index', range(P['conditional']['c3']['size']))
        keys = [k for k in w30[0] if 'CPU' not in k]
        res['conditional']['c3']['v27CURmismatches'] = [r['index'] for r in w30 if any(r[k] != v27[r['index']][k] for k in keys)]
    (HERE / 'results.json').write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1)[:3000])

if __name__ == '__main__':
    main()
