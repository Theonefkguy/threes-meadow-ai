"""V24 analysis (from repo root) -> results.json"""
from pathlib import Path
from math import comb, sqrt, erf
import json, statistics as S
R = Path('training/v24-acceptance'); pr = json.loads((R/'protocol.json').read_text())
A = {a: {r['seed']: r for r in map(json.loads, (R/f'games-{a}.jsonl').read_text().splitlines())} for a in ('OLD', 'NEW')}
seeds = sorted(A['OLD']); assert seeds == sorted(A['NEW']) and len(seeds) == pr['sampleSize']
Z = 1.959963984540054; n = len(seeds)
def wilson(k): p = k/n; d = 1+Z*Z/n; c = (p+Z*Z/(2*n))/d; h = Z*sqrt(p*(1-p)/n+Z*Z/(4*n*n))/d; return [c-h, c+h]
def mcn(b, c): m = b+c; return 1.0 if m == 0 else min(1.0, 2*sum(comb(m, i) for i in range(min(b, c)+1))/2**m)
out = {'arms': {}, 'primary': {}, 'secondary': {}}
for a, r in A.items():
    v = [r[s] for s in seeds]; e = {}
    for m in ('1536', '3072', '6144', '12288'):
        k = sum(x['r'+m] for x in v); e['reach'+m] = k; e['rate'+m] = k/n; e['wilson'+m] = wilson(k)
    e['meanScore'] = S.mean(x['score'] for x in v); e['medianScore'] = S.median(x['score'] for x in v)
    e['meanMoves'] = S.mean(x['moves'] for x in v); e['msPerMove'] = 1000*sum(x['searchCPU'] for x in v)/sum(x['moves'] for x in v)
    e['maxMoveMs'] = 1000*max(x['maxMoveCPU'] for x in v); e['cpuPerGame'] = S.mean(x['searchCPU'] for x in v)
    dh = [sum(x['depthHist'][i] for x in v) for i in range(6)]; e['depthCompletedShare'] = [d/sum(dh) for d in dh]
    out['arms'][a] = e
def cmp(m):
    on = sum(A['NEW'][s]['r'+m] and not A['OLD'][s]['r'+m] for s in seeds); oo = sum(A['OLD'][s]['r'+m] and not A['NEW'][s]['r'+m] for s in seeds)
    return {'diffPP': 100*(on-oo)/n, 'onlyNEW': on, 'onlyOLD': oo, 'p': mcn(on, oo)}
d = [A['NEW'][s]['score']-A['OLD'][s]['score'] for s in seeds]; m = sum(d)/n; se = sqrt(sum((x-m)**2 for x in d)/(n-1))/sqrt(n); z = m/se
P = {'P1_6144': cmp('6144'), 'P2_12288': cmp('12288'), 'P3_score': {'diff': m, 'ci95': [m-Z*se, m+Z*se], 'p': 2*(1-0.5*(1+erf(abs(z)/sqrt(2))))}}
order = sorted(P, key=lambda k: P[k]['p']); run = 0
for j, k in enumerate(order): run = max(run, min(1, (len(order)-j)*P[k]['p'])); P[k]['holmP'] = run
out['primary'] = P; out['secondary'] = {'1536': cmp('1536'), '3072': cmp('3072')}
(R/'results.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
