"""V39 analysis -> results.json: where do W31 games from the first 6144 fail? (descriptive)"""
import json, statistics as st
from pathlib import Path
L = Path('training/v39-endgame')
d = {r['index']: r for f in (L / 'games').glob('d-*.jsonl') for r in map(json.loads, f.read_text().splitlines())}
v38 = {r['index']: r for f in Path('training/v38-progress/eval').glob('W31-*.jsonl') for r in map(json.loads, f.read_text().splitlines())}
rows = [d[i] for i in range(2048)]
assert all(d[i][k] == v38[i][k] for i in range(2048) for k in ('success', 'moves', 'scoreGained')), 'replay differs from V38'
n = len(rows); S = ['start', 'm3072', 'm1536', 'm768', 'ready']
def reached(r, m): return m == 'start' or r[m] >= 0
funnel = {}
for m in S:
    g = [r for r in rows if reached(r, m)]; w = sum(r['success'] for r in g)
    funnel[m] = {'games': len(g), 'shareOfStates': len(g) / n, 'success': w, 'P12288': w / len(g) if g else None}
def last(r):
    for m in reversed(S):
        if reached(r, m): return m
fails = [r for r in rows if not r['success']]
lastStage = {m: sum(last(r) == m for r in fails) for m in S}
def q(v): v = sorted(v); return {'n': len(v), 'median': st.median(v), 'p10': v[len(v) // 10], 'p90': v[9 * len(v) // 10], 'max': v[-1]} if v else None
res = {'states': n, 'replayIdenticalToV38': True, 'funnel': funnel,
       'failuresByLastMilestone': lastStage,
       'movesFromLastMilestoneToDeath': {m: q([r['moves'] - (0 if m == 'start' else r[m]) for r in fails if last(r) == m]) for m in S},
       'successMovesReadyTo12288': q([r['moves'] - r['ready'] for r in rows if r['success']]),
       'successMovesM768To12288': q([r['moves'] - r['m768'] for r in rows if r['success'] and r['m768'] >= 0]),
       'failedAfterReady': q([r['moves'] - r['ready'] for r in rows if not r['success'] and r['ready'] >= 0]),
       'chain768': {k: {'games': len(g), 'success': sum(r['success'] for r in g), 'P12288': sum(r['success'] for r in g) / len(g) if g else None}
                    for k, g in (('chained', [r for r in rows if r['chain768'] == 1]), ('notChained', [r for r in rows if r['chain768'] == 0]))},
       'successWithoutM768': sum(r['success'] and r['m768'] < 0 for r in rows)}
(L / 'results.json').write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))
