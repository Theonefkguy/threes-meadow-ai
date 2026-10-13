"""V31 runner (repository root). Checkpointed; rerun any step to resume. pipeline.sh runs all.

python3 training/v31-stages/run.py pools          # stage-1 training and evaluation first-1536 pools
python3 training/v31-stages/run.py sample         # stage-1 on-policy training starts
python3 training/v31-stages/run.py freeze-pools
python3 training/v31-stages/run.py train S1|S3
python3 training/v31-stages/run.py compose        # W31 = W30 + new stage-1 and stage-3 tables/residuals; hashes
python3 training/v31-stages/run.py eval c1|c3|games
python3 training/v31-stages/analyze.py
Binaries: training/v31-stages/bin/{tools,train,game} (protocol.json compile).
"""
from pathlib import Path
import concurrent.futures as cf
import hashlib, json, os, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA, POOLS, MODELS, EVAL, BIN = HERE / 'data', HERE / 'pools', HERE / 'models', HERE / 'eval', HERE / 'bin'
P = json.loads((HERE / 'protocol.json').read_text())
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
ENV = dict(os.environ); ENV.pop('THREES_BONUS_RULE', None); ENV['CLAMP'] = '1'
W30BIN, W30RES = ROOT / P['base']['bin'], ROOT / P['base']['res']
STAGE_BYTES = 8 * 65536 * 4

def check():
    assert sha(W30BIN) == P['base']['binSHA256'] and sha(W30RES) == P['base']['resSHA256'], 'W30 hash'
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
    out = [tuple(P['pools'][name]['openings'])]
    for f in sorted(DATA.glob(f'{name}-extra-*.json')):
        out.append(tuple(json.loads(f.read_text())))
    return out

def opening_jobs(name):
    return [(name, s, min(25, last - s + 1)) for first, last in blocks(name) for s in range(first, last + 1, 25)]

def kept(name):
    return sorted((l for _, s, _ in opening_jobs(name) for l in (DATA / f'{name}-{s}.txt').read_text().splitlines()), key=lambda l: int(l.split()[0]))

def pools():
    check(); DATA.mkdir(exist_ok=True)
    def work(job):
        name, seed, n = job; out, log = DATA / f'{name}-{seed}.txt', DATA / f'{name}-{seed}.log.jsonl'
        if log.exists() and len(log.read_text().splitlines()) == n:
            return
        tmp, tlog = out.with_suffix('.part'), log.with_suffix('.part'); tmp.unlink(missing_ok=True); tlog.unlink(missing_ok=True)
        subprocess.run([str(BIN / 'tools'), 'pool', str(W30BIN), str(W30RES), str(seed), str(n), '12', str(tmp), str(tlog)], check=True, env=ENV, cwd=ROOT)
        if not tmp.exists():
            tmp.write_text('')
        assert [json.loads(l)['seed'] for l in tlog.read_text().splitlines()] == list(range(seed, seed + n))
        tmp.rename(out); tlog.rename(log)
    run_jobs([j for name in P['pools'] for j in opening_jobs(name)], work, 'pools')
    for name, spec in P['pools'].items():
        rows = kept(name); need = spec.get('size'); print(json.dumps({'pool': name, 'kept': len(rows), 'need': need}))
        if need and len(rows) < need:
            last = max(b[1] for b in blocks(name)); k = len(list(DATA.glob(f'{name}-extra-*.json'))) + 1
            (DATA / f'{name}-extra-{k}.json').write_text(json.dumps([last + 1, last + spec['extensionBlock']]))
            raise SystemExit('extension block written; rerun pools')

def sample():
    check(); POOLS.mkdir(exist_ok=True)
    src = POOLS / 'train1-first1536.txt'
    if not src.exists():
        src.write_text(''.join(r + '\n' for r in kept('train1')))
    n = len(src.read_text().splitlines()); tp = P['stage1Training']; jobs = list(range(0, n, 8))
    def work(s):
        out = DATA / f'sample-{s:04d}.txt'
        if out.exists():
            return
        tmp = out.with_suffix('.part'); tmp.unlink(missing_ok=True); tmp.write_text('')
        subprocess.run([str(BIN / 'tools'), 'sample', str(W30BIN), str(W30RES), str(src), str(s), str(min(8, n - s)), str(tp['sampleRolloutBase']),
                        str(tp['every']), '12', '13', str(tmp)], check=True, env=ENV, cwd=ROOT)
        tmp.rename(out)
    run_jobs(jobs, work, 'sample')
    (POOLS / 'train1-onpolicy.txt').write_text(''.join(l + '\n' for s in jobs for l in (DATA / f'sample-{s:04d}.txt').read_text().splitlines()))

def freeze_pools():
    path = HERE / 'pool-manifest.json'; assert not path.exists(), 'already frozen'
    rows = kept('eval1'); need = P['pools']['eval1']['size']; assert len(rows) >= need
    f = POOLS / 'eval1-first1536.txt'; f.write_text(''.join(r + '\n' for r in rows[:need]))
    logs = [json.loads(l) for _, s, _ in opening_jobs('eval1') for l in (DATA / f'eval1-{s}.log.jsonl').read_text().splitlines()]
    out = {'frozenUTC': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'eval1': {'blocks': blocks('eval1'), 'openings': len(logs), 'reached1536': sum(l['reached'] for l in logs), 'kept': len(rows), 'used': need,
                     'lastUsedSeed': int(rows[need - 1].split()[0]), 'sha256': sha(f)},
           'train1': {'first1536': len((POOLS / 'train1-first1536.txt').read_text().splitlines()), 'onpolicy': len((POOLS / 'train1-onpolicy.txt').read_text().splitlines()),
                      'train1-first1536.txt': sha(POOLS / 'train1-first1536.txt'), 'train1-onpolicy.txt': sha(POOLS / 'train1-onpolicy.txt')}}
    path.write_text(json.dumps(out, indent=2) + '\n'); print(json.dumps(out))

def train(arm):
    check(); MODELS.mkdir(exist_ok=True); a = P['arms'][arm]
    pool = ROOT / a['pool']
    if arm == 'S1':
        assert sha(pool) == json.loads((HERE / 'pool-manifest.json').read_text())['train1']['train1-onpolicy.txt']
    else:
        assert sha(pool) == a['poolSHA256']
    prefix = MODELS / arm; assert not Path(f'{prefix}-final.bin').exists(), 'already trained'
    subprocess.run([str(BIN / 'train'), str(W30BIN), f"{W30RES},zero:{a['stage']}", str(a['stage']), str(P['training']['threads']), str(a['moves']), str(a['seed']),
                    str(HERE / f'train-{arm}.log.jsonl'), str(P['training']['strongMoveShare']), ','.join(map(str, a['checkpoints'])), str(prefix), str(pool)],
                   check=True, cwd=ROOT)

def compose():
    path = HERE / 'models-manifest.json'; assert not path.exists(), 'already frozen'
    w30 = W30BIN.read_bytes(); s1 = (MODELS / 'S1-final.bin').read_bytes(); s3 = (MODELS / 'S3-final.bin').read_bytes()
    seg = lambda raw, s: raw[16 + s * STAGE_BYTES:16 + (s + 1) * STAGE_BYTES]
    assert len(w30) == len(s1) == len(s3) == 16 + 4 * STAGE_BYTES and w30[:16] == s1[:16] == s3[:16]
    for s in (0, 2, 3):
        assert seg(s1, s) == seg(w30, s), f'S1 changed stage {s}'
    for s in (0, 1, 2):
        assert seg(s3, s) == seg(w30, s), f'S3 changed stage {s}'
    out = w30[:16] + seg(w30, 0) + seg(s1, 1) + seg(w30, 2) + seg(s3, 3)
    (MODELS / 'W31.bin').write_bytes(out)
    rec = {'W31.bin': sha(MODELS / 'W31.bin'), 'S1-final.res': sha(MODELS / 'S1-final.res'), 'S3-final.res': sha(MODELS / 'S3-final.res'),
           'W30.bin': sha(W30BIN), 'W30.res': sha(W30RES)}
    path.write_text(json.dumps({'frozenUTC': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'sha256': rec,
                                'W31': 'W30 stages 0 and 2 + S1 stage 1 + S3 stage 3; residuals W30 stage 2, S1 stage 1, S3 stage 3',
                                'note': 'recorded after training, before any evaluation run'}, indent=2) + '\n'); print(json.dumps(rec))

def arms():
    return {'W30': (W30BIN, str(W30RES)), 'W31': (MODELS / 'W31.bin', f"{W30RES},{MODELS / 'S1-final.res'},{MODELS / 'S3-final.res'}")}

def evaluate(which):
    check(); EVAL.mkdir(exist_ok=True)
    rec = json.loads((HERE / 'models-manifest.json').read_text())['sha256']
    assert sha(MODELS / 'W31.bin') == rec['W31.bin'] and sha(MODELS / 'S1-final.res') == rec['S1-final.res'] and sha(MODELS / 'S3-final.res') == rec['S3-final.res']
    A = arms()
    if which == 'games':
        first, last = P['games']['seeds']
        def work(job):
            arm, s = job; n = min(16, last - s + 1); out = EVAL / f'games-{arm}-{s}.jsonl'
            if out.exists() and len(out.read_text().splitlines()) == n:
                return
            tmp = out.with_suffix('.part'); tmp.unlink(missing_ok=True); b, r = A[arm]
            subprocess.run([str(BIN / 'game'), str(b), r, '5', '0.01', '0', '1', str(s), str(n), str(tmp)], check=True, env=ENV, cwd=ROOT)
            assert [json.loads(l)['seed'] for l in tmp.read_text().splitlines()] == list(range(s, s + n))
            tmp.rename(out)
        run_jobs([(a, s) for s in range(first, last + 1, 16) for a in A], work, 'games'); return
    spec = P['conditional'][which]; pool = ROOT / spec['pool']
    assert sha(pool) == (json.loads((HERE / 'pool-manifest.json').read_text())['eval1']['sha256'] if which == 'c1' else spec['poolSHA256'])
    def work(job):
        arm, s = job; count = min(16, spec['size'] - s); out = EVAL / f'{which}-{arm}-{s:04d}.jsonl'
        if out.exists() and len(out.read_text().splitlines()) == count:
            return
        tmp = out.with_suffix('.part'); tmp.unlink(missing_ok=True); b, r = A[arm]
        subprocess.run([str(BIN / 'tools'), 'cont', str(b), r, '5', '0.01', str(pool), str(s), str(count), str(spec['rolloutBase']), str(spec['target']), str(tmp)],
                       check=True, env=ENV, cwd=ROOT)
        assert [json.loads(l)['index'] for l in tmp.read_text().splitlines()] == list(range(s, s + count))
        tmp.rename(out)
    run_jobs([(a, s) for s in range(0, spec['size'], 16) for a in A], work, which)

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'pools': pools()
    elif cmd == 'sample': sample()
    elif cmd == 'freeze-pools': freeze_pools()
    elif cmd == 'train': train(sys.argv[2])
    elif cmd == 'compose': compose()
    elif cmd == 'eval': evaluate(sys.argv[2])
    else: raise SystemExit(__doc__)
