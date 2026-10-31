"""V55: one round's fine-tune (fit.py) on a rented AutoDL GPU (adl, the autodl skill), called by fit.py.
Stage folder on the external SSD: fit.py, train.py, model.py, models.py, the init weights and the data
(positions, human positions, DAgger rounds 1..ROUND). Then: rent (4090D, else 4090 48G, else 4080S 32G; not the 5090: torch 2.4.1 for the image's Python 3.8 has no kernels for it) ->
push -> install torch 2.4.1 -> fit.py there -> a watch process (adl watch-start, independent of this
one) pulls runs/rROUND back and releases the instance. Resumable: the phase is kept in
runs/rROUND/cloud.json, so after an lq restart this only waits for the watch again.
A failed run leaves the instance switched off (disk kept) and this exits with an error."""
import json, shutil, subprocess, sys, time
from pathlib import Path
L = Path(__file__).resolve().parent; T = L.parent
ADL = str(Path.home() / '.claude/skills/autodl/bin/adl'); WATCH = Path.home() / '.config/adl'
DATA = Path('/Volumes/Extreme SSD/Datasets/threes-v50')
TORCH = 'python -m venv /root/t && /root/t/bin/pip install -q numpy torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121'

def adl(*a):
    print('adl', *a, flush=True); r = subprocess.run([ADL, *a], capture_output=True, text=True); print(r.stdout + r.stderr, flush=True)
    if r.returncode: raise SystemExit(f'adl {a[0]} failed')
    return r.stdout

def stage(R, init):
    s = DATA / 'cloud' / f'v55-r{R}'
    if s.exists(): shutil.rmtree(s)
    (s / 'data' / 'dagger').mkdir(parents=True)
    for f in (L / 'fit.py', T / 'v50-transformer' / 'train.py', T / 'v50-transformer' / 'model.py', T / 'v54-symmetry' / 'models.py'): shutil.copy(f, s)
    shutil.copy(init, s / 'init.state')                           # adl push leaves out *.pt
    for f in ('positions.npz', 'human.npz'): shutil.copy(DATA / f, s / 'data')
    for r in range(1, R + 1): shutil.copy(DATA / 'dagger' / f'round-{r}.npz', s / 'data' / 'dagger')
    return s

def train(R, init):
    run = L / 'runs' / f'r{R}'; run.mkdir(parents=True, exist_ok=True); st = run / 'cloud.json'
    name, job, rdir = f'v55-r{R}', f'v55-fit-r{R}', f'/root/v55-r{R}'
    phase = json.loads(st.read_text())['phase'] if st.exists() else None
    save = lambda p: st.write_text(json.dumps({'phase': p, 'name': name, 'job': job, 'time': time.strftime('%F %T')}) + '\n')
    if phase is None:
        s = stage(R, init); t0 = time.time()
        save('creating')
        try: adl('create', name, '--spec', '4090D,v-48g,v-32g-p', '--wait', '60')
        except SystemExit: st.unlink(); raise                     # nothing rented: start over next time
        save('created')
        adl('push', name, str(s)); print(f'pushed in {time.time() - t0:.0f} s', flush=True); phase = 'pushed'
    elif phase == 'creating':
        raise SystemExit(f'{st}: interrupted while renting; check adl list, release a leftover adl-{name} and delete {st}')
    if phase in ('created', 'pushed'):
        if phase == 'created': adl('push', name, str(DATA / 'cloud' / f'v55-r{R}'))
        adl('run', name, job, '--dir', rdir, '--out', f'runs/r{R}', '--', f'{TORCH} && V55_DATA=data /root/t/bin/python fit.py {R} init.state')
        adl('watch-start', name, job, '--dir', rdir, '--out', f'runs/r{R}', '--to', str(run), '--require', 'model.pt', '--max-hours', '4')
        save('running')
    log = WATCH / f'watch_{job}.log'
    while True:                                                   # the watch process pulls the results and releases
        text = log.read_text() if log.exists() else ''
        if '完成，结果已拉回' in text and (run / 'model.pt').exists():
            save('done'); print(text, flush=True); shutil.rmtree(DATA / 'cloud' / f'v55-r{R}', ignore_errors=True); return
        if '关机' in text or '看护退出' in text:
            print(text, flush=True); raise SystemExit(f'cloud round {R} failed; see {run}/adl_job.log and adl list')
        time.sleep(60)
