"""V27 analysis, fixed before any outcome (see analysis-pre-results-manifest.json).

python3 training/v27-stage3/analyze.py screen | interim | final    (repository root)
Writes screen.json, confirm-interim.json or confirm-final.json. Standard library only.
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
    """Two-sided exact McNemar on paired booleans: a-only vs b-only."""
    ao = sum(x and not y for x, y in zip(a, b)); bo = sum(y and not x for x, y in zip(a, b)); n = ao + bo
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(ao, bo) + 1)) / 2 ** n)
    return ao, bo, p

def paired(x, y):
    """Paired mean difference x-y, normal-approximation 95% CI and two-sided p (n >= 1024)."""
    d = [a - b for a, b in zip(x, y)]; n = len(d); m = sum(d) / n
    se = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1)) / math.sqrt(n)
    z = m / se if se else 0.0
    return {'meanDifference': m, 'approx95': [m - Z * se, m + Z * se], 'z': z, 'p': math.erfc(abs(z) / math.sqrt(2))}

def load(tag, arm, n, seeds):
    rows = {}
    for f in (HERE / 'eval').glob(f'{tag}-{arm}-*.jsonl'):
        for l in f.read_text().splitlines():
            r = json.loads(l); rows[r['index']] = r
    rows = [rows[i] for i in range(n)]
    assert [r['sourceSeed'] for r in rows] == seeds[:n], f'{tag} {arm} source seeds'
    assert all(r['depth'] == 5 and abs(r['threshold'] - 0.01) < 1e-12 and r['startRank'] == 14 for r in rows)
    assert all(r['success'] == (r['finalRank'] >= 15) and (r['success'] or r['died']) for r in rows)
    return rows

def arm_stats(rows):
    n = len(rows); k = sum(r['success'] for r in rows)
    return {'n': n, 'reach12288': k, 'rate': k / n, 'wilson95': wilson(k, n),
            'meanScoreGained': sum(r['scoreGained'] for r in rows) / n,
            'meanMoves': sum(r['moves'] for r in rows) / n,
            'avoidableCertainDeath': sum(r['lastKind'] == 2 for r in rows),
            'cpuMsPerMove': 1000 * sum(r['searchCPU'] for r in rows) / sum(r['moves'] for r in rows)}

def compare(x, y):
    ao, bo, p = mcnemar([r['success'] for r in x], [r['success'] for r in y])
    return {'candidateOnly': ao, 'baselineOnly': bo, 'differencePP': 100 * (ao - bo) / len(x), 'mcnemarExactP': p,
            'scoreGained': paired([r['scoreGained'] for r in x], [r['scoreGained'] for r in y])}

def seeds_of(path):
    return [int(l.split()[0]) for l in (ROOT / path).read_text().splitlines()]

def screen():
    sp = P['screen']; n = sp['size']; seeds = seeds_of(sp['pool'])
    g = {a: load('screen', a, n, seeds) for a in ('A', 'B', 'CUR')}
    a, b = sum(r['success'] for r in g['A']), sum(r['success'] for r in g['B'])
    sa, sb = (sum(r['scoreGained'] for r in g[x]) for x in ('A', 'B'))
    winner = 'B' if (b > a or (b == a and sb > sa)) else 'A'
    # Reproduction check (descriptive): CUR here is the V26 arm B setting on the same pool and seeds.
    v26 = [json.loads(l) for l in (ROOT / sp['v26Baseline']).read_text().splitlines()]
    keys = [k for k in v26[0] if k not in ('searchCPU', 'maxMoveCPU')]
    mismatch = [r['index'] for r, s in zip(g['CUR'], v26) if any(r[k] != s[k] for k in keys)]
    res = {'winner': winner, 'rule': sp['rule'], 'arms': {x: arm_stats(g[x]) for x in g},
           'descriptive': {'A vs CUR': compare(g['A'], g['CUR']), 'B vs CUR': compare(g['B'], g['CUR']), 'B vs A': compare(g['B'], g['A'])},
           'v26Reproduction': {'pairs': n, 'mismatchedIndices': mismatch}}
    (HERE / 'screen.json').write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))

def confirm(phase):
    winner = json.loads((HERE / 'screen.json').read_text())['winner']
    c = P['confirm']; seeds = seeds_of('training/v27-stage3/pools/confirm6144.txt')
    n = c['interimPairs'] if phase == 'interim' else P['confirmPool']['size']
    if phase == 'final' and json.loads((HERE / 'confirm-interim.json').read_text())['stoppedEarly']:
        n = c['interimPairs']
    w, cur = load('confirm', winner, n, seeds), load('confirm', 'CUR', n, seeds)
    cmp = compare(w, cur); gain = cmp['candidateOnly'] > cmp['baselineOnly']
    res = {'phase': phase, 'winner': winner, 'pairs': n, 'arms': {winner: arm_stats(w), 'CUR': arm_stats(cur)}, 'primary': cmp}
    if phase == 'interim':
        res['stoppedEarly'] = cmp['mcnemarExactP'] < c['interimStopP']
    else:
        sc = cmp['scoreGained']
        res['gate'] = {'primarySignificantGain': gain and cmp['mcnemarExactP'] < c['alpha'],
                       'scoreNotSignificantlyWorse': not (sc['p'] < 0.05 and sc['meanDifference'] < 0)}
        res['nativeGatesPass'] = all(res['gate'].values())
        res['note'] = 'website adoption additionally requires the browser parity/timing checks in protocol.adoption'
    (HERE / f'confirm-{phase}.json').write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))

if __name__ == '__main__':
    phase = sys.argv[1]
    if phase == 'screen': screen()
    elif phase in ('interim', 'final'): confirm(phase)
    else: raise SystemExit(__doc__)
