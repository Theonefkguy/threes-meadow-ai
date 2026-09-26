"""V22 runs (from repo root). Build: g++ -std=c++17 -O3 -march=native training/v21-late-train/snap.cpp -o /tmp/v21-audit"""
from pathlib import Path
import subprocess, os, concurrent.futures as cf
R = Path('training/v22-clamp'); out = R/'runs'; out.mkdir(parents=True, exist_ok=True); V = Path('training/v21-late-train')
M = {'CURc': 'training/v7/base.ntd', 'NEWc': str(V/'models/NEW.bin')}
E = {'E1': (V/'pools/EVAL3072.txt', 14, 21710001, 1024), 'E2': (V/'pools/EVAL6144.txt', 15, 21720001, 512)}
jobs = [(e, m, i) for e in E for i in range(0, E[e][3], 64) for m in M]
def run(j):
    e, m, i = j; pool, t, base, n = E[e]; f = out/f'{e}-{m}-{i:04d}.jsonl'
    if f.exists(): return
    tmp = out/f'{e}-{m}-{i:04d}.part'; tmp.unlink(missing_ok=True)
    subprocess.run(['/tmp/v21-audit','cont',M[m],'5','0.01',str(pool),str(i),'64',str(base),str(t),str(tmp)], check=True, env={**os.environ,'CLAMP':'1'})
    tmp.rename(f); print('done', e, m, i, flush=True)
with cf.ThreadPoolExecutor(2) as ex: list(ex.map(run, jobs))
print('complete', flush=True)
