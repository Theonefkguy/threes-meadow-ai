"""V28 analysis (repository root): python3 training/v28-curve/analyze.py -> results.json
Fixed before any checkpoint outcome. CUR and the final model (= V27 arm A) reuse the V27
continuations on the same pools and seeds. All comparisons are descriptive (exploratory round).
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
    return ao, bo, p

def paired(x, y):
    d = [a - b for a, b in zip(x, y)]; n = len(d); m = sum(d) / n
    se = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1)) / math.sqrt(n)
    return {'meanDifference': m, 'approx95': [m - Z * se, m + Z * se]}

def rows_from(glob_dir, pattern, n, seeds):
    rows = {}
    for f in glob_dir.glob(pattern):
        for l in f.read_text().splitlines():
            r = json.loads(l); rows[r['index']] = r
    rows = [rows[i] for i in range(n)]
    assert [r['sourceSeed'] for r in rows] == seeds
    assert all(r['depth'] == 5 and abs(r['threshold'] - 0.01) < 1e-12 and r['startRank'] == 14 for r in rows)
    return rows

def load(arm):
    out = []
    for pool in P['pools']:
        seeds = [int(l.split()[0]) for l in (ROOT / pool['path']).read_text().splitlines()]
        if arm in ('CUR', 'FINAL'):
            src = pool['reuse'][arm]; out += rows_from(ROOT / src['dir'], src['pattern'], pool['size'], seeds)
        else:
            out += rows_from(HERE / 'eval', f"{pool['name']}-{arm}-*.jsonl", pool['size'], seeds)
    return out

def main():
    arms = ['CUR'] + [f'C{c}' for c in P['checkpoints']] + ['FINAL']
    moves = {'CUR': 0, 'FINAL': P['training']['moves'], **{f'C{c}': c for c in P['checkpoints']}}
    g = {a: load(a) for a in arms}; n = len(g['CUR']); sizes = [p['size'] for p in P['pools']]
    res = {'pairs': n, 'curve': []}
    for a in arms:
        win = [r['success'] for r in g[a]]; k = sum(win)
        per_pool, at = {}, 0
        for pool, s in zip(P['pools'], sizes):
            per_pool[pool['name']] = sum(win[at:at + s]); at += s
        item = {'arm': a, 'trainingMoves': moves[a], 'reach12288': k, 'rate': k / n, 'wilson95': wilson(k, n), 'perPool': per_pool,
                'meanScoreGained': sum(r['scoreGained'] for r in g[a]) / n, 'avoidableCertainDeath': sum(r['lastKind'] == 2 for r in g[a])}
        if a != 'CUR':
            ao, bo, p = mcnemar(win, [r['success'] for r in g['CUR']])
            item['vsCUR'] = {'armOnly': ao, 'curOnly': bo, 'mcnemarExactP_unadjusted': p,
                             'scoreGained': paired([r['scoreGained'] for r in g[a]], [r['scoreGained'] for r in g['CUR']])}
        res['curve'].append(item)
    cur = res['curve'][0]['reach12288']; ck = res['curve'][1:-1]
    res['readout'] = {
        'everyCheckpointBelowCUR': all(c['reach12288'] < cur for c in ck),
        'checkpointsAboveCUR': [c['arm'] for c in ck if c['reach12288'] > cur],
        'rule': P['readout']}
    (HERE / 'results.json').write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))

if __name__ == '__main__':
    main()
