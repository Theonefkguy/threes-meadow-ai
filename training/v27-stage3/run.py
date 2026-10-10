"""V27 runner (repository root). Checkpointed; every step can be rerun to resume.

python3 training/v27-stage3/run.py pool-train  training/v27-stage3/bin/pool    # training first-6144 states
python3 training/v27-stage3/run.py sample      training/v27-stage3/bin/sample  # on-policy training starts
python3 training/v27-stage3/run.py pool-eval   training/v27-stage3/bin/pool    # confirmation first-6144 states
python3 training/v27-stage3/run.py freeze-eval
python3 training/v27-stage3/run.py train ARM   training/v27-stage3/bin/train   # ARM = A or B
python3 training/v27-stage3/run.py freeze-models training/v27-stage3/bin/compose
python3 training/v27-stage3/run.py screen          training/v27-stage3/bin/snap  # then: analyze.py screen
python3 training/v27-stage3/run.py confirm-interim training/v27-stage3/bin/snap  # then: analyze.py interim
python3 training/v27-stage3/run.py confirm-final   training/v27-stage3/bin/snap  # then: analyze.py final
"""
from pathlib import Path
import concurrent.futures as cf
import hashlib, json, os, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA, POOLS, MODELS, EVAL = HERE / 'data', HERE / 'pools', HERE / 'models', HERE / 'eval'
P = json.loads((HERE / 'protocol.json').read_text())
CUR = ROOT / P['baseModel']['path']
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
ENV = dict(os.environ); ENV.pop('THREES_BONUS_RULE', None); ENV['CLAMP'] = '1'

def check(binary, kind):
    assert sha(CUR) == P['baseModel']['sha256'], 'base model hash'
    for f, h in P['sourceSHA256'].items():
        assert sha(ROOT / f) == h, f'source hash {f}'
    assert sha(binary) == P['binarySHA256'][kind], f'{kind} binary hash: compile as in the protocol'

def run_jobs(jobs, work, label, workers=None):
    start = time.time()
    with cf.ThreadPoolExecutor(max_workers=workers or P['workers']) as pool:
        futures = {pool.submit(work, j) for j in jobs}
        total = len(futures)
        while futures:
            done, futures = cf.wait(futures, timeout=60, return_when=cf.FIRST_EXCEPTION)
            for f in done:
                f.result()
            finished = total - len(futures); el = time.time() - start
            print(json.dumps({'stage': label, 'jobsDone': finished, 'jobs': total, 'elapsedSeconds': round(el),
                              'etaSeconds': round(el / finished * (total - finished)) if finished else None}), flush=True)

def openings(binary, blocks, tag):
    """Website default AI from fresh openings to the first 6144 (training/v26-late-cutoff/pool.cpp)."""
    check(binary, 'pool'); DATA.mkdir(exist_ok=True)
    chunk = 25
    jobs = [(s, min(chunk, last - s + 1)) for first, last in blocks for s in range(first, last + 1, chunk)]
    def work(job):
        seed, n = job
        out, log = DATA / f'{tag}-{seed}.txt', DATA / f'{tag}-{seed}.log.jsonl'
        if log.exists() and len(log.read_text().splitlines()) == n:
            return
        tmp, tlog = out.with_suffix('.part'), log.with_suffix('.part')
        tmp.unlink(missing_ok=True); tlog.unlink(missing_ok=True)
        subprocess.run([str(binary), 'pool', str(CUR), str(seed), str(n), str(tmp), str(tlog)], check=True, env=ENV, cwd=ROOT)
        if not tmp.exists():
            tmp.write_text('')
        assert [json.loads(l)['seed'] for l in tlog.read_text().splitlines()] == list(range(seed, seed + n))
        tmp.rename(out); tlog.rename(log)
    run_jobs(jobs, work, tag)
    rows, logs = [], []
    for seed, n in jobs:
        rows += (DATA / f'{tag}-{seed}.txt').read_text().splitlines()
        logs += (DATA / f'{tag}-{seed}.log.jsonl').read_text().splitlines()
    rows.sort(key=lambda l: int(l.split()[0]))
    return rows, [json.loads(l) for l in logs]

def eval_blocks():
    blocks = [tuple(P['confirmPool']['initialBlock'])]
    for f in sorted(DATA.glob('eval-extra-*.json')):
        blocks.append(tuple(json.loads(f.read_text())))
    return blocks

def pool_train(binary):
    rows, logs = openings(binary, [tuple(P['trainingPool']['openings'])], 'train')
    POOLS.mkdir(exist_ok=True)
    (POOLS / 'train-first6144.txt').write_text(''.join(r + '\n' for r in rows))
    print(json.dumps({'openings': len(logs), 'reached6144': sum(l['reached6144'] for l in logs), 'kept': len(rows)}))

def sample(binary):
    check(binary, 'sample')
    src = POOLS / 'train-first6144.txt'; n = len(src.read_text().splitlines()); chunk = 8
    tp = P['trainingPool']
    jobs = list(range(0, n, chunk))
    def work(s):
        count = min(chunk, n - s)
        out, log = DATA / f'sample-{s:04d}.txt', DATA / f'sample-{s:04d}.log.jsonl'
        if log.exists() and len(log.read_text().splitlines()) == count:
            return
        tmp, tlog = out.with_suffix('.part'), log.with_suffix('.part')
        tmp.unlink(missing_ok=True); tlog.unlink(missing_ok=True)
        subprocess.run([str(binary), 'sample', str(CUR), str(src), str(s), str(count), str(tp['sampleRolloutBase']), str(tp['every']), str(tmp), str(tlog)],
                       check=True, env=ENV, cwd=ROOT)
        if not tmp.exists():
            tmp.write_text('')
        assert [json.loads(l)['index'] for l in tlog.read_text().splitlines()] == list(range(s, s + count))
        tmp.rename(out); tlog.rename(log)
    run_jobs(jobs, work, 'sample')
    rows = [l for s in jobs for l in (DATA / f'sample-{s:04d}.txt').read_text().splitlines()]
    logs = [json.loads(l) for s in jobs for l in (DATA / f'sample-{s:04d}.log.jsonl').read_text().splitlines()]
    (POOLS / 'train-onpolicy.txt').write_text(''.join(r + '\n' for r in rows))
    (HERE / 'training-pool.json').write_text(json.dumps({
        'firstStates': n, 'onpolicyStates': len(rows), 'rolloutMoves': sum(l['moves'] for l in logs),
        'train-first6144.txt': sha(src), 'train-onpolicy.txt': sha(POOLS / 'train-onpolicy.txt')}, indent=2) + '\n')
    print(json.dumps({'onpolicyStates': len(rows)}))

def pool_eval(binary):
    rows, logs = openings(binary, eval_blocks(), 'eval')
    need = P['confirmPool']['size']
    (HERE / 'eval-pool-log.jsonl').write_text(''.join(json.dumps(l) + '\n' for l in sorted(logs, key=lambda l: l['seed'])))
    print(json.dumps({'openings': len(logs), 'reached6144': sum(l['reached6144'] for l in logs), 'kept': len(rows), 'need': need}))
    if len(rows) < need:
        last = max(b[1] for b in eval_blocks()); k = len(list(DATA.glob('eval-extra-*.json'))) + 1
        ext = [last + 1, last + P['confirmPool']['extensionBlock']]
        (DATA / f'eval-extra-{k}.json').write_text(json.dumps(ext))
        print('fewer than needed; extension block written, rerun pool-eval:', ext)

def freeze_eval():
    blocks = eval_blocks(); need = P['confirmPool']['size']
    rows = sorted((l for first, last in blocks for s in range(first, last + 1, 25) for l in (DATA / f'eval-{s}.txt').read_text().splitlines()),
                  key=lambda l: int(l.split()[0]))
    assert len(rows) >= need, 'run pool-eval until enough states'
    POOLS.mkdir(exist_ok=True); out = POOLS / 'confirm6144.txt'
    assert not out.exists(), 'already frozen'
    out.write_text(''.join(r + '\n' for r in rows[:need]))
    logs = [json.loads(l) for l in (HERE / 'eval-pool-log.jsonl').read_text().splitlines()]
    manifest = {'frozenUTC': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'blocks': blocks, 'openings': len(logs),
                'reached6144': sum(l['reached6144'] for l in logs), 'terminalExcluded': sum(l['reached6144'] and not l['kept'] for l in logs),
                'kept': len(rows), 'used': need, 'lastUsedSeed': int(rows[need - 1].split()[0]),
                'confirm6144.txt': sha(out), 'eval-pool-log.jsonl': sha(HERE / 'eval-pool-log.jsonl')}
    (HERE / 'eval-pool-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))

def train(arm, binary):
    check(binary, 'train')
    tp = json.loads((HERE / 'training-pool.json').read_text())
    assert sha(POOLS / 'train-onpolicy.txt') == tp['train-onpolicy.txt']
    a = P['arms'][arm]; MODELS.mkdir(exist_ok=True)
    out = MODELS / f'{arm}-train.bin'
    assert not out.exists(), 'already trained'
    subprocess.run([str(binary), str(CUR), str(out), str(a['moves']), str(a['seed']), str(HERE / f'train-{arm}.log.jsonl'),
                    str(P['training']['strongMoveShare']), str(POOLS / 'train-onpolicy.txt')], check=True, cwd=ROOT)
    print('trained', arm, sha(out))

def freeze_models(binary):
    check(binary, 'compose')
    out = {'CUR': sha(CUR)}
    for arm in P['arms']:
        subprocess.run([str(binary), str(CUR), str(MODELS / f'{arm}-train.bin'), str(MODELS / f'{arm}.bin')], check=True, cwd=ROOT)
        out[arm] = sha(MODELS / f'{arm}.bin')
    path = HERE / 'models-manifest.json'
    assert not path.exists(), 'already frozen'
    path.write_text(json.dumps({'frozenUTC': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'sha256': out,
                                'note': 'recorded after training, before any evaluation run'}, indent=2) + '\n')
    print(json.dumps(out))

def model_path(arm):
    return CUR if arm == 'CUR' else MODELS / f'{arm}.bin'

def continuations(binary, tag, pool, base, arms, n):
    check(binary, 'snap')
    frozen = json.loads((HERE / 'models-manifest.json').read_text())['sha256']
    for arm in arms:
        assert sha(model_path(arm)) == frozen[arm], f'model {arm} changed after freeze'
    EVAL.mkdir(exist_ok=True); chunk = 16
    jobs = [(arm, s) for s in range(0, n, chunk) for arm in arms]
    def work(job):
        arm, s = job; count = min(chunk, n - s)
        out = EVAL / f'{tag}-{arm}-{s:04d}.jsonl'
        if out.exists() and len(out.read_text().splitlines()) == count:
            return
        tmp = out.with_suffix('.part'); tmp.unlink(missing_ok=True)
        subprocess.run([str(binary), 'cont', str(model_path(arm)), '5', '0.01', str(pool), str(s), str(count), str(base), '15', str(tmp)],
                       check=True, env=ENV, cwd=ROOT)
        assert [json.loads(l)['index'] for l in tmp.read_text().splitlines()] == list(range(s, s + count))
        tmp.rename(out)
    run_jobs(jobs, work, tag)

def screen(binary):
    sp = P['screen']; v26 = ROOT / sp['pool']
    assert sha(v26) == json.loads((ROOT / sp['poolManifest']).read_text())['pool6144.txt']
    continuations(binary, 'screen', v26, sp['rolloutBase'], ['A', 'B', 'CUR'], sp['size'])

def confirm(binary, phase):
    winner = json.loads((HERE / 'screen.json').read_text())['winner']
    cp = P['confirmPool']; pool = POOLS / 'confirm6144.txt'
    assert sha(pool) == json.loads((HERE / 'eval-pool-manifest.json').read_text())['confirm6144.txt']
    n = P['confirm']['interimPairs']
    if phase == 'final':
        assert (HERE / 'confirm-interim.json').exists(), 'interim analysis first'
        if json.loads((HERE / 'confirm-interim.json').read_text())['stoppedEarly']:
            print('stopped at interim; nothing to run'); return
        n = cp['size']
    continuations(binary, 'confirm', pool, P['confirm']['rolloutBase'], [winner, 'CUR'], n)

if __name__ == '__main__':
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == 'pool-train': pool_train(Path(args[0]))
    elif cmd == 'sample': sample(Path(args[0]))
    elif cmd == 'pool-eval': pool_eval(Path(args[0]))
    elif cmd == 'freeze-eval': freeze_eval()
    elif cmd == 'train': train(args[0], Path(args[1]))
    elif cmd == 'freeze-models': freeze_models(Path(args[0]))
    elif cmd == 'screen': screen(Path(args[0]))
    elif cmd == 'confirm-interim': confirm(Path(args[0]), 'interim')
    elif cmd == 'confirm-final': confirm(Path(args[0]), 'final')
    else: raise SystemExit(__doc__)
