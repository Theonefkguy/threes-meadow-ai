"""Summarize V17 games and state probes -> results.json (run from repo root)."""
from pathlib import Path
from math import comb, sqrt
import json, statistics as S
root = Path('training/v17-fast-depth'); pr = json.loads((root/'protocol.json').read_text())
def load(a): return {r['seed']: r for r in map(json.loads, (root/f'games-{a}.jsonl').read_text().splitlines())}
arms = {a: load(a) for a in pr['arms']}
seeds = sorted(arms['A']); assert all(sorted(arms[a]) == seeds for a in arms) and len(seeds) == pr['sampleSize']
def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, p)
out = {'arms': {}, 'comparisons': {}}
for a, rows in arms.items():
    r = [rows[s] for s in seeds]; succ = sum(x['success'] for x in r); moves = sum(x['moves'] for x in r)
    cpu = sum(x['searchCPU'] for x in r)
    out['arms'][a] = {**pr['arms'][a], 'success': succ, 'games': len(r), 'rate': succ/len(r),
        'meanMoveMs': 1000*cpu/moves, 'meanGameCPU': cpu/len(r),
        'p95MoveMsMedianGame': 1000*S.median(x['p95MoveCPU'] for x in r),
        'maxMoveMs': 1000*max(x['maxMoveCPU'] for x in r)}
for x, y in [('B', 'A'), ('C', 'A'), ('D', 'A'), ('D', 'B'), ('B', 'C')]:
    d = [int(arms[x][s]['success']) - int(arms[y][s]['success']) for s in seeds]
    only_x = sum(v == 1 for v in d); only_y = sum(v == -1 for v in d); n = len(d)
    mean = sum(d)/n; sd = sqrt(sum((v-mean)**2 for v in d)/(n-1)); half = 1.959963984540054*sd/sqrt(n)
    out['comparisons'][f'{x}-{y}'] = {'diffPP': 100*mean, 'ci95PP': [100*(mean-half), 100*(mean+half)],
        f'only{x}': only_x, f'only{y}': only_y, 'mcnemarP': mcnemar(only_x, only_y)}
(root/'results.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False))
