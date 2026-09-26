"""Death audit (run from repo root): replays the V21 evaluation continuations with the same
seeds and search, recording for each game whether the final move was a certain death with no
alternative (lastKind 1) or although a safer move existed (lastKind 2), and how often a move
with a higher immediate-death probability than the best legal alternative was chosen."""
from pathlib import Path
import subprocess, concurrent.futures as cf, json
R = Path('training/v21-late-train'); out = R/'audit'; out.mkdir(exist_ok=True)
M = {'CUR': 'training/v7/base.ntd', 'NEW': str(R/'models/NEW.bin')}
E = {'E1': (R/'pools/EVAL3072.txt', 14, 21710001, 1024), 'E2': (R/'pools/EVAL6144.txt', 15, 21720001, 512)}
jobs = [(e, m, i) for e in E for i in range(0, E[e][3], 64) for m in M]
def run(j):
    e, m, i = j; pool, t, base, n = E[e]; f = out/f'{e}-{m}-{i:04d}.jsonl'
    if f.exists(): return
    tmp = out/f'{e}-{m}-{i:04d}.part'; tmp.unlink(missing_ok=True)
    subprocess.run(['/tmp/v21-audit','cont',M[m],'5','0.01',str(pool),str(i),'64',str(base),str(t),str(tmp)], check=True); tmp.rename(f); print('done', e, m, i, flush=True)
with cf.ThreadPoolExecutor(2) as ex: list(ex.map(run, jobs))
