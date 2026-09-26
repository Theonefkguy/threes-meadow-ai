"""V22 analysis (from repo root) -> training/v22-clamp/results.json"""
from pathlib import Path
from math import comb, sqrt, erf
import json
R = Path('training/v22-clamp'); A = Path('training/v21-late-train/audit')
N = {'E1': 1024, 'E2': 512}
def load(d, e, m): return {x['index']: x for f in d.glob(f'{e}-{m}-*.jsonl') for x in map(json.loads, f.read_text().splitlines())}
arms = {e: {'CUR': load(A, e, 'CUR'), 'CURc': load(R/'runs', e, 'CURc'), 'NEWc': load(R/'runs', e, 'NEWc')} for e in N}
for e in N:
    for m, r in arms[e].items(): assert sorted(r) == list(range(N[e])), (e, m, len(r))
def mcn(b, c): n = b+c; return 1.0 if n == 0 else min(1.0, 2*sum(comb(n, i) for i in range(min(b, c)+1))/2**n)
def paired(d):
    n = len(d); m = sum(d)/n; se = sqrt(sum((x-m)**2 for x in d)/(n-1))/sqrt(n); z = m/se if se else 0
    return m, [m-1.96*se, m+1.96*se], 2*(1-0.5*(1+erf(abs(z)/sqrt(2))))
def cmp_success(e, x, y):
    a, b = arms[e][x], arms[e][y]; ox = sum(a[i]['success'] and not b[i]['success'] for i in a); oy = sum(b[i]['success'] and not a[i]['success'] for i in a)
    return {'diffPP': 100*(ox-oy)/N[e], 'onlyFirst': ox, 'onlySecond': oy, 'p': mcn(ox, oy)}
def cmp_gain(e, x, y):
    a, b = arms[e][x], arms[e][y]; m, ci, p = paired([a[i]['scoreGained']-b[i]['scoreGained'] for i in sorted(a)])
    return {'diff': m, 'ci95': ci, 'p': p}
out = {'arms': {}, 'primary': {}, 'secondary': {}}
for e in N:
    for m, r in arms[e].items():
        v = list(r.values()); died = [x for x in v if x['died']]
        out['arms'][f'{e}-{m}'] = {'n': len(v), 'success': sum(x['success'] for x in v), 'meanGain': sum(x['scoreGained'] for x in v)/len(v),
            'second3072': sum(x['maxSecondRank'] >= 13 for x in v), 'died': len(died), 'avoidableDeath': sum(x['lastKind'] == 2 for x in died),
            'forcedDeath': sum(x['lastKind'] == 1 for x in died), 'meanMoves': sum(x['moves'] for x in v)/len(v),
            'msPerMove': 1000*sum(x['searchCPU'] for x in v)/sum(x['moves'] for x in v)}
P = {'P1': cmp_success('E1', 'CURc', 'CUR'), 'P2': cmp_gain('E2', 'CURc', 'CUR'), 'P3': cmp_success('E1', 'NEWc', 'CURc'), 'P4': cmp_gain('E2', 'NEWc', 'CURc')}
order = sorted(P, key=lambda k: P[k]['p']); run = 0
for j, k in enumerate(order): run = max(run, min(1, (len(order)-j)*P[k]['p'])); P[k]['holmP'] = run
out['primary'] = P
out['secondary'] = {'E1 NEWc-CUR success': cmp_success('E1', 'NEWc', 'CUR'), 'E2 NEWc-CUR gain': cmp_gain('E2', 'NEWc', 'CUR'),
                    'E2 12288 CURc-CUR': cmp_success('E2', 'CURc', 'CUR'), 'E2 12288 NEWc-CURc': cmp_success('E2', 'NEWc', 'CURc'),
                    'E1 gain NEWc-CURc': cmp_gain('E1', 'NEWc', 'CURc')}
(R/'results.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
