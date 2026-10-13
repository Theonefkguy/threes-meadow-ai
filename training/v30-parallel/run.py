"""V30 runner (repository root). Checkpointed; rerun any step to resume.

python3 training/v30-parallel/run.py train S      # 3e9-move parallel replica of V29 R (Hogwild check)
python3 training/v30-parallel/run.py train P      # 1.2e10-move parallel run with checkpoints
python3 training/v30-parallel/run.py freeze-models
python3 training/v30-parallel/run.py select       # new candidates on the V29 curve pool
python3 training/v30-parallel/analyze.py select   # writes select.json (winner)
python3 training/v30-parallel/run.py games        # whole games, CUR vs winner
python3 training/v30-parallel/analyze.py games
Binaries: training/v30-parallel/bin/{train,game}; training/v29-five/bin/tools (see protocol.json).
"""
from pathlib import Path
import concurrent.futures as cf
import hashlib, json, os, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
V29 = ROOT / 'training/v29-five'
MODELS, EVAL, BIN = HERE / 'models', HERE / 'eval', HERE / 'bin'
P = json.loads((HERE / 'protocol.json').read_text())
CUR = ROOT / P['baseModel']['path']
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
ENV = dict(os.environ); ENV.pop('THREES_BONUS_RULE', None); ENV['CLAMP'] = '1'

def check():
    assert sha(CUR) == P['baseModel']['sha256'], 'base model hash'
    for f, h in P['sourceSHA256'].items():
        assert sha(ROOT / f) == h, f'source hash {f}'
    for f, h in P['binarySHA256'].items():
        assert sha(ROOT / f) == h, f'binary hash {f}'

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

def train(arm):
    check(); MODELS.mkdir(exist_ok=True)
    assert sha(V29 / 'pools/train-onpolicy.txt') == P['training']['poolSHA256']
    a = P['arms'][arm]; prefix = MODELS / arm
    assert not Path(f'{prefix}-final.bin').exists(), 'already trained'
    subprocess.run([str(ROOT / 'training/v30-parallel/bin/train'), str(CUR), 'zero', str(P['training']['threads']), str(a['moves']), str(a['seed']),
                    str(HERE / f'train-{arm}.log.jsonl'), str(P['training']['strongMoveShare']), ','.join(map(str, a['checkpoints'])), str(prefix),
                    str(V29 / 'pools/train-onpolicy.txt')], check=True, cwd=ROOT)
    print('trained', arm)

def files(tag):
    """CUR, V29R (V29 R-final), or ARM-CHECKPOINT of this round."""
    if tag == 'CUR':
        return CUR, '-'
    if tag == 'V29R':
        return V29 / 'models/R-final.bin', V29 / 'models/R-final.res'
    arm, ck = tag.split('-', 1)
    return MODELS / f'{arm}-{ck}.bin', MODELS / f'{arm}-{ck}.res'

def frozen_ok(tag):
    if tag == 'CUR':
        return
    b, r = files(tag)
    if tag == 'V29R':
        rec = json.loads((V29 / 'models-manifest.json').read_text())['sha256']['R-final']
        assert {b.name: sha(b), r.name: sha(r)} == rec, 'V29 R-final changed'
        return
    assert json.loads((HERE / 'models-manifest.json').read_text())['sha256'][tag] == {b.name: sha(b), r.name: sha(r)}, f'{tag} changed'

def freeze_models():
    path = HERE / 'models-manifest.json'; assert not path.exists(), 'already frozen'
    cur = CUR.read_bytes(); stage = (len(cur) - 16) // 4; rec = {}
    for tag in P['candidates']:
        if tag == 'V29R':
            continue
        b, r = files(tag); raw = b.read_bytes()
        assert raw[:16] == cur[:16] and len(raw) == len(cur)
        for s in (0, 1, 3):  # only stage 2 may change
            assert raw[16 + s * stage:16 + (s + 1) * stage] == cur[16 + s * stage:16 + (s + 1) * stage], f'{tag} stage {s} changed'
        rec[tag] = {b.name: sha(b), r.name: sha(r)}
    path.write_text(json.dumps({'frozenUTC': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'sha256': rec,
                                'note': 'recorded after training, before any evaluation run; stages 0, 1 and 3 verified identical to CUR'}, indent=2) + '\n')
    print(json.dumps(rec))

def select():
    check(); EVAL.mkdir(exist_ok=True)
    sp = P['select']; pool = V29 / 'pools/curve3072.txt'
    assert sha(pool) == json.loads((V29 / 'pool-manifest.json').read_text())['curve']['sha256']
    tags = [t for t in P['candidates'] if t != 'V29R']
    for t in tags:
        frozen_ok(t)
    def work(job):
        t, s = job; count = min(16, sp['size'] - s); out = EVAL / f'select-{t}-{s:04d}.jsonl'
        if out.exists() and len(out.read_text().splitlines()) == count:
            return
        tmp = out.with_suffix('.part'); tmp.unlink(missing_ok=True); b, r = files(t)
        subprocess.run([str(V29 / 'bin/tools'), 'cont', str(b), str(r), '5', '0.01', str(pool), str(s), str(count), str(sp['rolloutBase']), '14', str(tmp)],
                       check=True, env=ENV, cwd=ROOT)
        assert [json.loads(l)['index'] for l in tmp.read_text().splitlines()] == list(range(s, s + count))
        tmp.rename(out)
    run_jobs([(t, s) for s in range(0, sp['size'], 16) for t in tags], work, 'select')

def games():
    check(); EVAL.mkdir(exist_ok=True)
    winner = json.loads((HERE / 'select.json').read_text())['winner']; frozen_ok(winner)
    first, last = P['games']['seeds']; chunk = 16
    arms = {'CUR': files('CUR'), 'W': files(winner)}
    def work(job):
        arm, s = job; n = min(chunk, last - s + 1); out = EVAL / f'games-{arm}-{s}.jsonl'
        if out.exists() and len(out.read_text().splitlines()) == n:
            return
        tmp = out.with_suffix('.part'); tmp.unlink(missing_ok=True); b, r = arms[arm]
        subprocess.run([str(ROOT / 'training/v30-parallel/bin/game'), str(b), str(r), '5', '0.01', '0', '1', str(s), str(n), str(tmp)], check=True, env=ENV, cwd=ROOT)
        assert [json.loads(l)['seed'] for l in tmp.read_text().splitlines()] == list(range(s, s + n))
        tmp.rename(out)
    run_jobs([(a, s) for s in range(first, last + 1, chunk) for a in arms], work, 'games')

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'train': train(sys.argv[2])
    elif cmd == 'freeze-models': freeze_models()
    elif cmd == 'select': select()
    elif cmd == 'games': games()
    else: raise SystemExit(__doc__)
