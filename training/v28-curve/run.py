"""V28 runner (repository root). Checkpointed; rerun any step to resume.

python3 training/v28-curve/run.py train   training/v28-curve/bin/train   # replays V27 arm A, saving checkpoints
python3 training/v28-curve/run.py compose training/v27-stage3/bin/compose # each checkpoint that exists; hash recorded once
python3 training/v28-curve/run.py eval    training/v27-stage3/bin/snap    # every composed checkpoint, both pools
python3 training/v28-curve/analyze.py
"""
from pathlib import Path
import concurrent.futures as cf
import hashlib, json, os, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
V27 = ROOT / 'training/v27-stage3'
MODELS, EVAL = HERE / 'models', HERE / 'eval'
P = json.loads((HERE / 'protocol.json').read_text())
CUR = ROOT / P['baseModel']['path']
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
ENV = dict(os.environ); ENV.pop('THREES_BONUS_RULE', None); ENV['CLAMP'] = '1'
CHECKS = P['checkpoints']

def check(binary, kind):
    assert sha(CUR) == P['baseModel']['sha256'], 'base model hash'
    for f, h in P['sourceSHA256'].items():
        assert sha(ROOT / f) == h, f'source hash {f}'
    assert sha(binary) == P['binarySHA256'][kind], f'{kind} binary hash'

def train(binary):
    check(binary, 'train'); MODELS.mkdir(exist_ok=True)
    t = P['training']; final = MODELS / 'final-train.bin'
    assert not final.exists(), 'already trained'
    assert sha(V27 / 'pools/train-onpolicy.txt') == t['poolSHA256']
    subprocess.run([str(binary), str(CUR), str(final), str(t['moves']), str(t['seed']), str(HERE / 'train.log.jsonl'), str(t['strongMoveShare']),
                    ','.join(map(str, CHECKS)), str(MODELS / 'ckpt'), str(V27 / 'pools/train-onpolicy.txt')], check=True, cwd=ROOT)
    h = sha(final); print('final', h)
    assert h == t['expectedFinalSHA256'], 'final model differs from V27 arm A: trajectory not reproduced'

def compose(binary):
    check(binary, 'compose')
    mf = HERE / 'models-manifest.json'
    rec = json.loads(mf.read_text()) if mf.exists() else {}
    for c in CHECKS:
        src, out = MODELS / f'ckpt-{c}.bin', MODELS / f'C{c}.bin'
        if not src.exists() or f'C{c}' in rec:
            continue
        subprocess.run([str(binary), str(CUR), str(src), str(out)], check=True, cwd=ROOT)
        rec[f'C{c}'] = sha(out)
        mf.write_text(json.dumps(rec, indent=2) + '\n'); print('composed', c, rec[f'C{c}'])

def evaluate(binary):
    check(binary, 'snap')
    rec = json.loads((HERE / 'models-manifest.json').read_text()); EVAL.mkdir(exist_ok=True)
    arms = [a for a in (f'C{c}' for c in CHECKS) if a in rec]
    for a in arms:
        assert sha(MODELS / f'{a}.bin') == rec[a], f'model {a} changed'
    jobs = []
    for pool in P['pools']:
        assert sha(ROOT / pool['path']) == pool['sha256']
        jobs += [(pool, a, s) for s in range(0, pool['size'], 16) for a in arms]
    def work(job):
        pool, a, s = job; count = min(16, pool['size'] - s)
        out = EVAL / f"{pool['name']}-{a}-{s:04d}.jsonl"
        if out.exists() and len(out.read_text().splitlines()) == count:
            return
        tmp = out.with_suffix('.part'); tmp.unlink(missing_ok=True)
        subprocess.run([str(binary), 'cont', str(MODELS / f'{a}.bin'), '5', '0.01', str(ROOT / pool['path']), str(s), str(count),
                        str(pool['rolloutBase']), '15', str(tmp)], check=True, env=ENV, cwd=ROOT)
        assert [json.loads(l)['index'] for l in tmp.read_text().splitlines()] == list(range(s, s + count))
        tmp.rename(out)
    start = time.time()
    with cf.ThreadPoolExecutor(P['workers']) as ex:
        futures = {ex.submit(work, j) for j in jobs}; total = len(futures)
        while futures:
            done, futures = cf.wait(futures, timeout=60, return_when=cf.FIRST_EXCEPTION)
            for f in done:
                f.result()
            print(json.dumps({'arms': arms, 'jobsDone': total - len(futures), 'jobs': total, 'elapsedSeconds': round(time.time() - start)}), flush=True)

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'train': train(Path(sys.argv[2]))
    elif cmd == 'compose': compose(Path(sys.argv[2]))
    elif cmd == 'eval': evaluate(Path(sys.argv[2]))
    else: raise SystemExit(__doc__)
