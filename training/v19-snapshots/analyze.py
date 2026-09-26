"""Summarize V19 continuations -> results.json (run from repo root)."""
from pathlib import Path
from math import comb, sqrt
import json, statistics as S
root = Path('training/v19-snapshots'); pr = json.loads((root/'protocol.json').read_text())
arms = {a: {r['index']: r for r in map(json.loads, (root/f'cont-{a}.jsonl').read_text().splitlines())} for a in pr['arms']}
idx = sorted(arms['D5']); N = pr['pool']['size']; assert idx == list(range(N)) and all(sorted(arms[a]) == idx for a in arms)
Z = 1.959963984540054
def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n)
def wilson(k, n):
    p = k/n; d = 1 + Z*Z/n; c = (p + Z*Z/(2*n))/d; h = Z*sqrt(p*(1-p)/n + Z*Z/(4*n*n))/d
    return [c-h, c+h]
def compare(x, y):
    d = [int(arms[x][i]['success']) - int(arms[y][i]['success']) for i in idx]; n = len(d)
    ox, oy = d.count(1), d.count(-1); m = sum(d)/n; sd = sqrt(sum((v-m)**2 for v in d)/(n-1)); h = Z*sd/sqrt(n)
    return {'diffPP': 100*m, 'ci95PP': [100*(m-h), 100*(m+h)], 'onlyFirst': ox, 'onlySecond': oy, 'mcnemarP': mcnemar(ox, oy)}
out = {'arms': {}, 'primary': {}, 'exploratory': {}}
for a, rows in arms.items():
    r = [rows[i] for i in idx]; k = sum(x['success'] for x in r); moves = sum(x['moves'] for x in r); cpu = sum(x['searchCPU'] for x in r)
    out['arms'][a] = {**pr['arms'][a], 'success': k, 'n': N, 'rate': k/N, 'wilson95': wilson(k, N), 'meanMoveMs': 1000*cpu/moves,
                      'meanContCPU': cpu/N, 'maxMoveMs': 1000*max(x['maxMoveCPU'] for x in r), 'meanMovesSurvived': moves/N}
cands = [a for a in pr['arms'] if a != 'D5']
prim = {a: compare(a, 'D5') for a in cands}
order = sorted(cands, key=lambda a: prim[a]['mcnemarP']); running = 0
for j, a in enumerate(order):                      # Holm step-down adjusted p-values
    running = max(running, min(1.0, (len(order)-j) * prim[a]['mcnemarP'])); prim[a]['holmP'] = running
out['primary'] = {f'{a}-D5': prim[a] for a in cands}
for i, a in enumerate(cands):
    for b in cands[i+1:]: out['exploratory'][f'{a}-{b}'] = compare(a, b)
(root/'results.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False))
