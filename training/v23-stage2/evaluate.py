"""V23 screen / confirm (from repo root): python3 evaluate.py screen | confirm-interim | confirm-final"""
from pathlib import Path
import subprocess, os, sys, json, concurrent.futures as cf
from math import comb, sqrt, erf
R = Path('training/v23-stage2'); out = R/'eval'; out.mkdir(exist_ok=True)
def mcn(b, c): n = b+c; return 1.0 if n == 0 else min(1.0, 2*sum(comb(n, i) for i in range(min(b, c)+1))/2**n)
def run_jobs(pool, base, models, lo, hi, tag):
    jobs = [(m, i) for i in range(lo, hi, 64) for m in models]
    def run(j):
        m, i = j; f = out/f'{tag}-{m}-{i:04d}.jsonl'
        if f.exists(): return
        tmp = out/f'{tag}-{m}-{i:04d}.part'; tmp.unlink(missing_ok=True)
        subprocess.run(['/tmp/v21-audit','cont',str(R/f'models/{m}.bin'),'5','0.01',str(pool),str(i),str(min(64,hi-i)),str(base),'14',str(tmp)], check=True, env={**os.environ,'CLAMP':'1'})
        tmp.rename(f); print('done', tag, m, i, flush=True)
    with cf.ThreadPoolExecutor(2) as ex: list(ex.map(run, jobs))
def load(tag, m, hi): 
    r = {x['index']: x for f in out.glob(f'{tag}-{m}-*.jsonl') for x in map(json.loads, f.read_text().splitlines())}
    return {i: r[i] for i in range(hi)}
def compare(tag, x, y, hi):
    a, b = load(tag, x, hi), load(tag, y, hi); ox = sum(a[i]['success'] and not b[i]['success'] for i in a); oy = sum(b[i]['success'] and not a[i]['success'] for i in a)
    d = [a[i]['scoreGained']-b[i]['scoreGained'] for i in a]; m = sum(d)/len(d); se = sqrt(sum((v-m)**2 for v in d)/(len(d)-1))/sqrt(len(d))
    return {'n': hi, x: sum(v['success'] for v in a.values()), y: sum(v['success'] for v in b.values()), 'diffPP': 100*(ox-oy)/hi, f'only{x}': ox, f'only{y}': oy, 'p': mcn(ox, oy),
            'gainDiff': m, 'gainCI95': [m-1.96*se, m+1.96*se], 'avoidable': {x: sum(v['lastKind']==2 for v in a.values()), y: sum(v['lastKind']==2 for v in b.values())}}
phase = sys.argv[1]
if phase == 'screen':
    run_jobs(R/'pools/SCREEN.txt', 23310001, ['A', 'B'], 0, 512, 'screen')
    res = compare('screen', 'A', 'B', 512); res['winner'] = 'B' if res['B'] > res['A'] else 'A'
    (R/'screen.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
else:
    w = json.loads((R/'screen.json').read_text())['winner']; hi = 1280 if phase == 'confirm-interim' else 2560
    if phase == 'confirm-final' and (R/'stopped').exists(): hi = 1280
    run_jobs(R/'pools/CONFIRM.txt', 23320001, [w, 'CUR'], 0, hi, 'confirm')
    res = compare('confirm', w, 'CUR', hi); res['winner'] = w
    if phase == 'confirm-interim' and res['p'] < 0.001: (R/'stopped').write_text('Haybittle-Peto interim stop\n')
    (R/f'{phase}.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
