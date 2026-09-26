"""V21 paired evaluation per protocol.json (run from repo root): python3 .../evaluate.py [interim|final]
Builds nothing; expects /tmp/v21-snap and models/NEW.bin. Resumable chunks."""
from pathlib import Path
import subprocess, json, sys, concurrent.futures as cf
from math import comb, sqrt, erf
R = Path('training/v21-late-train'); pr = json.loads((R/'protocol.json').read_text())
MODELS = {'CUR': 'training/v7/base.ntd', 'NEW': str(R/'models/NEW.bin')}
E = {'E1': (R/'pools/EVAL3072.txt', 14, pr['evaluation']['E1']['rolloutBase']),
     'E2': (R/'pools/EVAL6144.txt', 15, pr['evaluation']['E2']['rolloutBase'])}
CH = 32; out = R/'eval'; out.mkdir(exist_ok=True)
def n_of(e): return len(E[e][0].read_text().splitlines())
def run(job):
    e, m, i = job; pool, target, base = E[e]; f = out/f'{e}-{m}-{i:04d}.jsonl'
    if f.exists(): return
    tmp = f.with_suffix('.tmp'); tmp.unlink(missing_ok=True)
    subprocess.run(['/tmp/v21-snap', 'cont', MODELS[m], '5', '0.01', str(pool), str(i), str(CH), str(base), str(target), str(tmp)], check=True)
    tmp.rename(f); print('done', e, m, i, flush=True)
def rows(e, m, upto):
    r = {}
    for f in out.glob(f'{e}-{m}-*.jsonl'):
        for l in f.read_text().splitlines():
            x = json.loads(l)
            if x['index'] < upto: r[x['index']] = x
    return r
def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2*sum(comb(n, i) for i in range(min(b, c)+1))/2**n)
def pt(d):  # two-sided paired t test via normal approx (n >= 256)
    n = len(d); m = sum(d)/n; sd = sqrt(sum((x-m)**2 for x in d)/(n-1)); z = m/(sd/sqrt(n)) if sd else 0
    return m, sd/sqrt(n), 2*(1-0.5*(1+erf(abs(z)/sqrt(2))))
def analyze(upto_frac):
    res = {}
    for e in E:
        frac = upto_frac[e] if isinstance(upto_frac, dict) else upto_frac
        N = int(n_of(e)*frac); a, b = rows(e, 'CUR', N), rows(e, 'NEW', N); idx = sorted(set(a) & set(b)); assert len(idx) == N, (e, len(idx), N)
        s = {'n': N}
        for m, r in (('CUR', a), ('NEW', b)):
            s[m] = {'success': sum(r[i]['success'] for i in idx), 'meanGain': sum(r[i]['scoreGained'] for i in idx)/N,
                    'second3072': sum(r[i]['maxSecondRank'] >= 13 for i in idx), 'meanMoves': sum(r[i]['moves'] for i in idx)/N,
                    'msPerMove': 1000*sum(r[i]['searchCPU'] for i in idx)/max(1, sum(r[i]['moves'] for i in idx))}
        on = sum(b[i]['success'] and not a[i]['success'] for i in idx); oc = sum(a[i]['success'] and not b[i]['success'] for i in idx)
        s['success'] = {'diffPP': 100*(on-oc)/N, 'onlyNEW': on, 'onlyCUR': oc, 'p': mcnemar(on, oc)}
        m, se, p = pt([b[i]['scoreGained'] - a[i]['scoreGained'] for i in idx]); s['gain'] = {'diff': m, 'ci95': [m-1.96*se, m+1.96*se], 'p': p}
        on = sum(b[i]['maxSecondRank'] >= 13 and a[i]['maxSecondRank'] < 13 for i in idx); oc = sum(a[i]['maxSecondRank'] >= 13 and b[i]['maxSecondRank'] < 13 for i in idx)
        s['second3072'] = {'onlyNEW': on, 'onlyCUR': oc, 'p': mcnemar(on, oc)}
        s['primaryP'] = s['success']['p'] if e == 'E1' else s['gain']['p']
        res[e] = s
    ps = sorted(res, key=lambda e: res[e]['primaryP']); run_ = 0
    for j, e in enumerate(ps): run_ = max(run_, min(1, (len(ps)-j)*res[e]['primaryP'])); res[e]['holmP'] = run_
    return res
phase = sys.argv[1] if len(sys.argv) > 1 else 'interim'
frac = 0.5 if phase == 'interim' else 1.0
todo = [e for e in E if not (phase == 'final' and (R/f'stopped-{e}').exists())]
jobs = [(e, m, i) for e in todo for i in range(0, int(n_of(e)*frac), CH) for m in MODELS]
with cf.ThreadPoolExecutor(2) as ex: list(ex.map(run, jobs))
if phase == 'interim':
    res = analyze(0.5); (R/'interim.json').write_text(json.dumps(res, indent=1))
    for e in E:
        if res[e]['primaryP'] < 0.001: (R/f'stopped-{e}').write_text('Haybittle-Peto interim stop\n')
    print(json.dumps(res, indent=1))
else:
    res = analyze({e: 0.5 if (R/f'stopped-{e}').exists() else 1.0 for e in E})
    ps = sorted(res, key=lambda e: res[e]['primaryP']); run_ = 0
    for j, e in enumerate(ps): run_ = max(run_, min(1, (len(ps)-j)*res[e]['primaryP'])); res[e]['holmP'] = run_
    (R/'results.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
