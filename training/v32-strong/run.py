"""V32 runner (repository root); pipeline.sh runs every step. Checkpointed; rerun to resume.

python3 training/v32-strong/run.py pool            # fresh first-6144 confirmation pool (W31 website search)
python3 training/v32-strong/run.py freeze-pool
python3 training/v32-strong/run.py train T3|T5     # run both at once (8 threads each)
python3 training/v32-strong/run.py freeze-models
python3 training/v32-strong/run.py screen          # T3, T5 on the V27 first-6144 pool (W31 reuses V31 c3)
python3 training/v32-strong/analyze.py screen
python3 training/v32-strong/run.py confirm         # winner and W31 on the fresh pool
python3 training/v32-strong/analyze.py confirm
"""
from pathlib import Path
import concurrent.futures as cf
import hashlib, json, os, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA, POOLS, MODELS, EVAL = HERE / 'data', HERE / 'pools', HERE / 'models', HERE / 'eval'
P = json.loads((HERE / 'protocol.json').read_text())
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
ENV = dict(os.environ); ENV.pop('THREES_BONUS_RULE', None); ENV['CLAMP'] = '1'
TOOLS, TRAIN = ROOT / 'training/v31-stages/bin/tools', HERE / 'bin/train'
B = P['base']; W31BIN = ROOT / B['bin']; RES = {k: ROOT / v for k, v in B['residuals'].items()}
STAGE_BYTES = 8 * 65536 * 4

def check():
    assert sha(W31BIN) == B['binSHA256'] and all(sha(RES[k]) == B['residualSHA256'][k] for k in RES), 'W31 hash'
    for f, h in P['sourceSHA256'].items():
        assert sha(ROOT / f) == h, f'source hash {f}'
    for f, h in P['binarySHA256'].items():
        assert sha(ROOT / f) == h, f'binary hash {f}'

def run_jobs(jobs, work, label, workers):
    start = time.time()
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(work, j) for j in jobs}; total = len(futures)
        while futures:
            done, futures = cf.wait(futures, timeout=60, return_when=cf.FIRST_EXCEPTION)
            for f in done:
                f.result()
            n = total - len(futures); el = time.time() - start
            print(json.dumps({'stage': label, 'jobsDone': n, 'jobs': total, 'elapsedSeconds': round(el),
                              'etaSeconds': round(el / n * (total - n)) if n else None}), flush=True)

def reslist(stage3):
    return ','.join(str(x) for x in (RES['2'], RES['1'], stage3))

def blocks():
    out = [tuple(P['pool']['openings'])]
    for f in sorted(DATA.glob('pool-extra-*.json')):
        out.append(tuple(json.loads(f.read_text())))
    return out

def opening_jobs():
    return [(s, min(25, last - s + 1)) for first, last in blocks() for s in range(first, last + 1, 25)]

def kept():
    return sorted((l for s, _ in opening_jobs() for l in (DATA / f'pool-{s}.txt').read_text().splitlines()), key=lambda l: int(l.split()[0]))

def pool():
    check(); DATA.mkdir(exist_ok=True)
    def work(job):
        seed, n = job; out, log = DATA / f'pool-{seed}.txt', DATA / f'pool-{seed}.log.jsonl'
        if log.exists() and len(log.read_text().splitlines()) == n:
            return
        tmp, tlog = out.with_suffix('.part'), log.with_suffix('.part'); tmp.unlink(missing_ok=True); tlog.unlink(missing_ok=True)
        subprocess.run([str(TOOLS), 'pool', str(W31BIN), reslist(RES['3']), str(seed), str(n), '14', str(tmp), str(tlog)], check=True, env=ENV, cwd=ROOT)
        if not tmp.exists():
            tmp.write_text('')
        assert [json.loads(l)['seed'] for l in tlog.read_text().splitlines()] == list(range(seed, seed + n))
        tmp.rename(out); tlog.rename(log)
    run_jobs(opening_jobs(), work, 'pool', P['workers'])
    rows = kept(); need = P['pool']['size']; print(json.dumps({'kept': len(rows), 'need': need}))
    if len(rows) < need:
        last = max(b[1] for b in blocks()); k = len(list(DATA.glob('pool-extra-*.json'))) + 1
        (DATA / f'pool-extra-{k}.json').write_text(json.dumps([last + 1, last + P['pool']['extensionBlock']]))
        raise SystemExit('extension block written; rerun pool')

def freeze_pool():
    path = HERE / 'pool-manifest.json'; assert not path.exists(), 'already frozen'
    rows = kept(); need = P['pool']['size']; assert len(rows) >= need
    POOLS.mkdir(exist_ok=True); f = POOLS / 'confirm6144.txt'; f.write_text(''.join(r + '\n' for r in rows[:need]))
    logs = [json.loads(l) for s, _ in opening_jobs() for l in (DATA / f'pool-{s}.log.jsonl').read_text().splitlines()]
    path.write_text(json.dumps({'frozenUTC': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'blocks': blocks(), 'openings': len(logs),
                                'reached6144': sum(l['reached'] for l in logs), 'kept': len(rows), 'used': need,
                                'lastUsedSeed': int(rows[need - 1].split()[0]), 'confirm6144.txt': sha(f)}, indent=2) + '\n')

def train(arm):
    check(); MODELS.mkdir(exist_ok=True); a = P['arms'][arm]; t = P['training']
    assert sha(ROOT / t['pool']) == t['poolSHA256']
    prefix = MODELS / arm; assert not Path(f'{prefix}-final.bin').exists(), 'already trained'
    subprocess.run([str(TRAIN), str(W31BIN), reslist(RES['3']), '3', str(t['threadsPerArm']), str(t['seconds']), str(a['seed']), str(HERE / f'train-{arm}.log.jsonl'),
                    '1', str(a['depth']), str(a['threshold']), ','.join(map(str, t['checkpointFractions'])), str(prefix), str(ROOT / t['pool'])], check=True, cwd=ROOT)

def freeze_models():
    path = HERE / 'models-manifest.json'; assert not path.exists(), 'already frozen'
    w = W31BIN.read_bytes(); seg = lambda raw, s: raw[16 + s * STAGE_BYTES:16 + (s + 1) * STAGE_BYTES]; rec = {}
    for arm in P['arms']:
        b = (MODELS / f'{arm}-final.bin').read_bytes(); assert b[:16] == w[:16] and len(b) == len(w)
        for s in (0, 1, 2):
            assert seg(b, s) == seg(w, s), f'{arm} changed stage {s}'
        rec[arm] = {f'{arm}-final.bin': sha(MODELS / f'{arm}-final.bin'), f'{arm}-final.res': sha(MODELS / f'{arm}-final.res')}
    path.write_text(json.dumps({'frozenUTC': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'sha256': rec,
                                'note': 'recorded after training, before any evaluation; stages 0-2 verified identical to W31'}, indent=2) + '\n')

def files(tag):
    if tag == 'W31':
        return W31BIN, reslist(RES['3'])
    rec = json.loads((HERE / 'models-manifest.json').read_text())['sha256'][tag]
    b, r = MODELS / f'{tag}-final.bin', MODELS / f'{tag}-final.res'
    assert {b.name: sha(b), r.name: sha(r)} == rec, f'{tag} changed after freeze'
    return b, reslist(r)

def cont(tag_list, pool_path, base, n, label):
    check(); EVAL.mkdir(exist_ok=True)
    F = {t: files(t) for t in tag_list}
    def work(job):
        t, s = job; count = min(16, n - s); out = EVAL / f'{label}-{t}-{s:04d}.jsonl'
        if out.exists() and len(out.read_text().splitlines()) == count:
            return
        tmp = out.with_suffix('.part'); tmp.unlink(missing_ok=True); b, r = F[t]
        subprocess.run([str(TOOLS), 'cont', str(b), r, '5', '0.01', str(pool_path), str(s), str(count), str(base), '15', str(tmp)], check=True, env=ENV, cwd=ROOT)
        assert [json.loads(l)['index'] for l in tmp.read_text().splitlines()] == list(range(s, s + count))
        tmp.rename(out)
    run_jobs([(t, s) for s in range(0, n, 16) for t in tag_list], work, label, P['workers'])

def screen():
    sp = P['screen']; assert sha(ROOT / sp['pool']) == sp['poolSHA256']
    cont(['T3', 'T5'], ROOT / sp['pool'], sp['rolloutBase'], sp['size'], 'screen')

def confirm():
    winner = json.loads((HERE / 'screen.json').read_text())['winner']
    pool_path = POOLS / 'confirm6144.txt'; assert sha(pool_path) == json.loads((HERE / 'pool-manifest.json').read_text())['confirm6144.txt']
    cont([winner, 'W31'], pool_path, P['confirm']['rolloutBase'], P['pool']['size'], 'confirm')

if __name__ == '__main__':
    cmd = sys.argv[1]
    {'pool': pool, 'freeze-pool': freeze_pool, 'freeze-models': freeze_models, 'screen': screen, 'confirm': confirm}.get(cmd, lambda: train(sys.argv[2]))()
