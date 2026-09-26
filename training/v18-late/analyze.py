"""Summarize V18 games -> results.json (run from repo root)."""
from pathlib import Path
from math import comb, sqrt
import json, statistics as S
root = Path('training/v18-late'); pr = json.loads((root/'protocol.json').read_text())
arms = {a: {r['seed']: r for r in map(json.loads, (root/f'games-{a}.jsonl').read_text().splitlines())} for a in pr['arms']}
seeds = sorted(arms['A']); assert all(sorted(arms[a]) == seeds for a in arms) and len(seeds) == pr['sampleSize']
Z = 1.959963984540054
def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n)
def wilson(k, n):
    p = k/n; d = 1 + Z*Z/n; c = (p + Z*Z/(2*n))/d; h = Z*sqrt(p*(1-p)/n + Z*Z/(4*n*n))/d
    return [c-h, c+h]
out = {'arms': {}, 'comparisons': {}}
for a, rows in arms.items():
    r = [rows[s] for s in seeds]; n = len(r); moves = sum(x['moves'] for x in r)
    e = {**pr['arms'][a], 'games': n}
    for m in ['1536', '3072', '6144']:
        k = sum(x['r'+m] for x in r); e['reach'+m] = k; e['rate'+m] = k/n; e['wilson'+m] = wilson(k, n)
    e['p6144given3072'] = e['reach6144']/e['reach3072'] if e['reach3072'] else None
    e['meanFinalScore'] = S.mean(x['finalScore'] for x in r); e['medianMoves'] = S.median(x['moves'] for x in r)
    cpu = [sum(x[k] for x in r) for k in ['cpuEarly', 'cpuMid', 'cpuLate']]
    e['meanMoveMs'] = 1000*sum(cpu)/moves; e['meanGameCPU'] = sum(cpu)/n; e['cpuShareByStage'] = [c/sum(cpu) for c in cpu]
    e['maxMoveMs'] = 1000*max(x['maxMoveCPU'] for x in r)
    out['arms'][a] = e
for x, y in [('B', 'A'), ('D', 'A'), ('D', 'B')]:
    for m in ['6144', '3072', '1536']:
        d = [int(arms[x][s]['r'+m]) - int(arms[y][s]['r'+m]) for s in seeds]; n = len(d)
        ox = d.count(1); oy = d.count(-1); mean = sum(d)/n; sd = sqrt(sum((v-mean)**2 for v in d)/(n-1)); h = Z*sd/sqrt(n)
        out['comparisons'][f'{x}-{y} reach{m}'] = {'diffPP': 100*mean, 'ci95PP': [100*(mean-h), 100*(mean+h)], f'only{x}': ox, f'only{y}': oy, 'mcnemarP': mcnemar(ox, oy)}
(root/'results.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False))
