"""V26 runner (repository root): pool -> freeze -> cont. Checkpointed, 8 workers.

python3 training/v26-late-cutoff/run.py pool   POOL_BINARY
python3 training/v26-late-cutoff/run.py freeze
python3 training/v26-late-cutoff/run.py cont   SNAP_BINARY
"""
from pathlib import Path
import concurrent.futures as cf
import hashlib, json, os, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
P = json.loads((HERE / 'protocol.json').read_text())
MODEL = ROOT / P['model']['path']
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
ENV = dict(os.environ); ENV.pop('THREES_BONUS_RULE', None); ENV['CLAMP'] = '1'

def check(binary, kind):
    assert sha(MODEL) == P['model']['sha256'], 'model hash'
    for f, h in P['sourceSHA256'].items():
        assert sha(ROOT / f) == h, f'source hash {f}'
    assert sha(binary) == P['binarySHA256'][kind], f'{kind} binary hash: compile and record a new protocol elsewhere'

def pool_blocks():
    first, last = P['pool']['initialBlock']
    blocks = [(first, last)]
    extra = sorted(DATA.glob('pool-extra-*.json'))
    for f in extra:
        blocks.append(tuple(json.loads(f.read_text())))
    return blocks

def run_jobs(jobs, work, label):
    start = time.time()
    with cf.ThreadPoolExecutor(max_workers=P['workers']) as pool:
        futures = {pool.submit(work, j) for j in jobs}
        total = len(futures)
        while futures:
            done, futures = cf.wait(futures, timeout=60, return_when=cf.FIRST_EXCEPTION)
            for f in done:
                f.result()
            finished = total - len(futures)
            el = time.time() - start
            print(json.dumps({'stage': label, 'jobsDone': finished, 'jobs': total, 'elapsedSeconds': round(el),
                              'etaSeconds': round(el / finished * (total - finished)) if finished else None}), flush=True)

def pool(binary):
    check(binary, 'pool'); DATA.mkdir(exist_ok=True)
    chunk = 25
    jobs = [(s, min(chunk, last - s + 1)) for first, last in pool_blocks() for s in range(first, last + 1, chunk)]
    def work(job):
        seed, n = job
        out, log = DATA / f'pool-{seed}.txt', DATA / f'pool-{seed}.log.jsonl'
        if log.exists() and len(log.read_text().splitlines()) == n:
            return
        tmp, tlog = out.with_suffix('.part'), log.with_suffix('.part')
        tmp.unlink(missing_ok=True); tlog.unlink(missing_ok=True)
        subprocess.run([str(binary), 'pool', str(MODEL), str(seed), str(n), str(tmp), str(tlog)], check=True, env=ENV, cwd=ROOT)
        if not tmp.exists():
            tmp.write_text('')
        assert [json.loads(l)['seed'] for l in tlog.read_text().splitlines()] == list(range(seed, seed + n))
        tmp.rename(out); tlog.rename(log)
    run_jobs(jobs, work, 'pool')
    kept = sum(len(f.read_text().splitlines()) for f in DATA.glob('pool-*.txt'))
    print(json.dumps({'keptStates': kept, 'needed': P['sampleSize']}))

def freeze():
    rows, logs = [], []
    for first, last in pool_blocks():
        for s in range(first, last + 1, 25):
            rows += (DATA / f'pool-{s}.txt').read_text().splitlines()
            logs += [json.loads(l) for l in (DATA / f'pool-{s}.log.jsonl').read_text().splitlines()]
    seeds = [int(r.split()[0]) for r in rows]
    assert seeds == sorted(seeds) and len(set(seeds)) == len(seeds)
    assert sum(l['kept'] for l in logs) == len(rows)
    if len(rows) < P['sampleSize']:
        raise SystemExit(f'only {len(rows)} kept states; add the next 400-seed block as data/pool-extra-N.json per protocol')
    out = HERE / 'pool6144.txt'
    out.write_text(''.join(r + '\n' for r in rows[:P['sampleSize']]))
    (HERE / 'pool-log.jsonl').write_text(''.join(json.dumps(l) + '\n' for l in logs))
    manifest = {'frozenUTC': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'openings': len(logs),
                'reached6144': sum(l['reached6144'] for l in logs), 'terminalExcluded': sum(l['reached6144'] and not l['kept'] for l in logs),
                'kept': len(rows), 'used': P['sampleSize'], 'lastUsedSeed': int(rows[P['sampleSize'] - 1].split()[0]),
                'pool6144.txt': sha(out), 'pool-log.jsonl': sha(HERE / 'pool-log.jsonl')}
    (HERE / 'pool-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))

def cont(binary):
    check(binary, 'snap')
    manifest = json.loads((HERE / 'pool-manifest.json').read_text())
    pool_file = HERE / 'pool6144.txt'
    assert sha(pool_file) == manifest['pool6144.txt'], 'pool changed after freeze'
    n, chunk = P['sampleSize'], 16
    arms = P['continuation']['arms']
    jobs = [(arm, s) for s in range(0, n, chunk) for arm in ('A2', 'A1', 'B')]
    def work(job):
        arm, s = job
        count = min(chunk, n - s)
        out = DATA / f'cont-{arm}-{s:04d}.jsonl'
        if out.exists() and len(out.read_text().splitlines()) == count:
            return
        tmp = out.with_suffix('.part'); tmp.unlink(missing_ok=True)
        a = arms[arm]
        subprocess.run([str(binary), 'cont', str(MODEL), str(a['depth']), str(a['threshold']), str(pool_file), str(s), str(count),
                        '26200001', '15', str(tmp)], check=True, env=ENV, cwd=ROOT)
        assert [json.loads(l)['index'] for l in tmp.read_text().splitlines()] == list(range(s, s + count))
        tmp.rename(out)
    run_jobs(jobs, work, 'cont')
    for arm in arms:
        rows = sorted((json.loads(l) for f in DATA.glob(f'cont-{arm}-*.jsonl') for l in f.read_text().splitlines()), key=lambda r: r['index'])
        assert [r['index'] for r in rows] == list(range(n))
        (HERE / f'games-{arm}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    print('continuations complete')

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'pool': pool(Path(sys.argv[2]))
    elif cmd == 'freeze': freeze()
    elif cmd == 'cont': cont(Path(sys.argv[2]))
    else: raise SystemExit(__doc__)
