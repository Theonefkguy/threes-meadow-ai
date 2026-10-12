"""V29 runner (repository root). Checkpointed; rerun any step to resume.

python3 training/v29-five/run.py pools                 # training, curve and confirmation first-3072 pools
python3 training/v29-five/run.py sample                # on-policy stage-2 training starts
python3 training/v29-five/run.py freeze-pools
python3 training/v29-five/run.py train C|R             # the two arms (run in parallel)
python3 training/v29-five/run.py freeze-models
python3 training/v29-five/run.py curve                 # checkpoints and finals on the curve pool
python3 training/v29-five/run.py confirm               # finals and CUR on the confirmation pool
python3 training/v29-five/analyze.py
Binaries: training/v29-five/bin/{tools,train} (see protocol.json compile).
"""
from pathlib import Path
import concurrent.futures as cf
import hashlib, json, os, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA, POOLS, MODELS, EVAL, BIN = HERE / 'data', HERE / 'pools', HERE / 'models', HERE / 'eval', HERE / 'bin'
P = json.loads((HERE / 'protocol.json').read_text())
CUR = ROOT / P['baseModel']['path']
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
ENV = dict(os.environ); ENV.pop('THREES_BONUS_RULE', None); ENV['CLAMP'] = '1'

def check():
    assert sha(CUR) == P['baseModel']['sha256'], 'base model hash'
    for f, h in P['sourceSHA256'].items():
        assert sha(ROOT / f) == h, f'source hash {f}'
    for k, h in P['binarySHA256'].items():
        assert sha(BIN / k) == h, f'{k} binary hash'

def run_jobs(jobs, work, label):
    start = time.time()
    with cf.ThreadPoolExecutor(max_workers=P['workers']) as ex:
        futures = {ex.submit(work, j) for j in jobs}; total = len(futures)
        while futures:
            done, futures = cf.wait(futures, timeout=60, return_when=cf.FIRST_EXCEPTION)
            for f in done:
                f.result()
            n = total - len(futures); el = time.time() - start
            print(json.dumps({'stage': label, 'jobsDone': n, 'jobs': total, 'elapsedSeconds': round(el),
                              'etaSeconds': round(el / n * (total - n)) if n else None}), flush=True)

def blocks(name):
    spec = P['pools'][name]; out = [tuple(spec['openings'])]
    for f in sorted(DATA.glob(f'{name}-extra-*.json')):
        out.append(tuple(json.loads(f.read_text())))
    return out

def opening_jobs(name):
    return [(name, s, min(25, last - s + 1)) for first, last in blocks(name) for s in range(first, last + 1, 25)]

def pools():
    check(); DATA.mkdir(exist_ok=True)
    def work(job):
        name, seed, n = job
        out, log = DATA / f'{name}-{seed}.txt', DATA / f'{name}-{seed}.log.jsonl'
        if log.exists() and len(log.read_text().splitlines()) == n:
            return
        tmp, tlog = out.with_suffix('.part'), log.with_suffix('.part'); tmp.unlink(missing_ok=True); tlog.unlink(missing_ok=True)
        subprocess.run([str(BIN / 'tools'), 'pool', str(CUR), '-', str(seed), str(n), '13', str(tmp), str(tlog)], check=True, env=ENV, cwd=ROOT)
        if not tmp.exists():
            tmp.write_text('')
        assert [json.loads(l)['seed'] for l in tlog.read_text().splitlines()] == list(range(seed, seed + n))
        tmp.rename(out); tlog.rename(log)
    run_jobs([j for name in P['pools'] for j in opening_jobs(name)], work, 'pools')
    for name, spec in P['pools'].items():
        rows = kept(name); need = spec.get('size')
        print(json.dumps({'pool': name, 'kept': len(rows), 'need': need}))
        if need and len(rows) < need:
            last = max(b[1] for b in blocks(name)); k = len(list(DATA.glob(f'{name}-extra-*.json'))) + 1
            (DATA / f'{name}-extra-{k}.json').write_text(json.dumps([last + 1, last + spec['extensionBlock']]))
            print('extension block written; rerun pools')

def kept(name):
    return sorted((l for _, s, _ in opening_jobs(name) for l in (DATA / f'{name}-{s}.txt').read_text().splitlines()), key=lambda l: int(l.split()[0]))

def sample():
    check(); POOLS.mkdir(exist_ok=True)
    src = POOLS / 'train-first3072.txt'
    if not src.exists():
        src.write_text(''.join(r + '\n' for r in kept('train')))
    n = len(src.read_text().splitlines()); tp = P['trainingPool']
    jobs = list(range(0, n, 8))
    def work(s):
        count = min(8, n - s); out = DATA / f'sample-{s:04d}.txt'
        if out.exists():
            return
        tmp = out.with_suffix('.part'); tmp.unlink(missing_ok=True); tmp.write_text('')
        subprocess.run([str(BIN / 'tools'), 'sample', str(CUR), '-', str(src), str(s), str(count), str(tp['sampleRolloutBase']), str(tp['every']), '13', '14', str(tmp)],
                       check=True, env=ENV, cwd=ROOT)
        tmp.rename(out)
    run_jobs(jobs, work, 'sample')
    (POOLS / 'train-onpolicy.txt').write_text(''.join(l + '\n' for s in jobs for l in (DATA / f'sample-{s:04d}.txt').read_text().splitlines()))

def freeze_pools():
    out = {}
    for name in ('curve', 'confirm'):
        rows = kept(name); need = P['pools'][name]['size']; assert len(rows) >= need, name
        f = POOLS / f'{name}3072.txt'; assert not f.exists(), 'already frozen'
        f.write_text(''.join(r + '\n' for r in rows[:need]))
        logs = [json.loads(l) for _, s, _ in opening_jobs(name) for l in (DATA / f'{name}-{s}.log.jsonl').read_text().splitlines()]
        out[name] = {'blocks': blocks(name), 'openings': len(logs), 'reached3072': sum(l['reached'] for l in logs), 'kept': len(rows), 'used': need,
                     'lastUsedSeed': int(rows[need - 1].split()[0]), 'sha256': sha(f)}
    out['train'] = {'first3072': len((POOLS / 'train-first3072.txt').read_text().splitlines()), 'onpolicy': len((POOLS / 'train-onpolicy.txt').read_text().splitlines()),
                    'train-first3072.txt': sha(POOLS / 'train-first3072.txt'), 'train-onpolicy.txt': sha(POOLS / 'train-onpolicy.txt')}
    out['frozenUTC'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (HERE / 'pool-manifest.json').write_text(json.dumps(out, indent=2) + '\n'); print(json.dumps(out))

def train(arm):
    check(); MODELS.mkdir(exist_ok=True)
    mf = json.loads((HERE / 'pool-manifest.json').read_text())
    assert sha(POOLS / 'train-onpolicy.txt') == mf['train']['train-onpolicy.txt']
    t = P['training']; prefix = MODELS / arm
    assert not Path(f'{prefix}-final.bin').exists(), 'already trained'
    subprocess.run([str(BIN / 'train'), str(CUR), P['arms'][arm]['res'], str(t['moves']), str(P['arms'][arm]['seed']), str(HERE / f'train-{arm}.log.jsonl'),
                    str(t['strongMoveShare']), ','.join(map(str, t['checkpoints'])), str(prefix), str(POOLS / 'train-onpolicy.txt')], check=True, cwd=ROOT)
    print('trained', arm)

def model_files(tag):
    """tag: CUR, or ARM-CHECKPOINT (e.g. R-final, C-1000000000)."""
    if tag == 'CUR':
        return [CUR], '-'
    arm, ck = tag.split('-', 1); b = MODELS / f'{arm}-{ck}.bin'
    return ([b], '-') if P['arms'][arm]['res'] == '-' else ([b, MODELS / f'{arm}-{ck}.res'], str(MODELS / f'{arm}-{ck}.res'))

def tags():
    return [f'{a}-{c}' for a in P['arms'] for c in [*map(str, P['training']['checkpoints']), 'final']]

def freeze_models():
    path = HERE / 'models-manifest.json'; assert not path.exists(), 'already frozen'
    rec = {t: {Path(f).name: sha(f) for f in model_files(t)[0]} for t in tags()}
    path.write_text(json.dumps({'frozenUTC': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'sha256': rec,
                                'note': 'recorded after training, before any evaluation run'}, indent=2) + '\n'); print(json.dumps(rec))

def evaluate(pool_name, which):
    check(); EVAL.mkdir(exist_ok=True)
    frozen = json.loads((HERE / 'models-manifest.json').read_text())['sha256']
    for t in which:
        if t != 'CUR':
            assert {Path(f).name: sha(f) for f in model_files(t)[0]} == frozen[t], f'{t} changed'
    pool = POOLS / f'{pool_name}3072.txt'; spec = P['pools'][pool_name]
    assert sha(pool) == json.loads((HERE / 'pool-manifest.json').read_text())[pool_name]['sha256']
    def work(job):
        t, s = job; count = min(16, spec['size'] - s); out = EVAL / f'{pool_name}-{t}-{s:04d}.jsonl'
        if out.exists() and len(out.read_text().splitlines()) == count:
            return
        tmp = out.with_suffix('.part'); tmp.unlink(missing_ok=True); files, res = model_files(t)
        subprocess.run([str(BIN / 'tools'), 'cont', str(files[0]), res, '5', '0.01', str(pool), str(s), str(count), str(spec['rolloutBase']), '14', str(tmp)],
                       check=True, env=ENV, cwd=ROOT)
        assert [json.loads(l)['index'] for l in tmp.read_text().splitlines()] == list(range(s, s + count))
        tmp.rename(out)
    run_jobs([(t, s) for s in range(0, spec['size'], 16) for t in which], work, pool_name)

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'pools': pools()
    elif cmd == 'sample': sample()
    elif cmd == 'freeze-pools': freeze_pools()
    elif cmd == 'train': train(sys.argv[2])
    elif cmd == 'freeze-models': freeze_models()
    elif cmd == 'curve': evaluate('curve', ['CUR', *tags()])
    elif cmd == 'confirm': evaluate('confirm', ['CUR', 'C-final', 'R-final'])
    else: raise SystemExit(__doc__)
