"""Fixed V19 runner (resumable). Build: g++ -std=c++17 -O3 -march=native training/v19-snapshots/snap.cpp -o /tmp/v19-snap"""
from pathlib import Path
import concurrent.futures, subprocess, json, hashlib, sys
root = Path('training/v19-snapshots'); pr = json.loads((root/'protocol.json').read_text())
assert hashlib.sha256(Path('training/v7/base.ntd').read_bytes()).hexdigest() == pr['baseSHA256']
pool = root/'pool3072.txt'; N = pr['pool']['size']; assert len(pool.read_text().splitlines()) == N
CH = 32; parts = root/'cont'; parts.mkdir(exist_ok=True); workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
jobs = [(a, i) for i in range(0, N, CH) for a in pr['arms']]
def run(job):
    arm, i = job; cfg = pr['arms'][arm]; out = parts/f'{arm}-{i:04d}.jsonl'
    if out.exists() and len(out.read_text().splitlines()) == CH: return
    tmp = out.with_suffix('.tmp'); tmp.unlink(missing_ok=True)
    subprocess.run(['/tmp/v19-snap', 'cont', str(cfg['depth']), str(cfg['threshold']), str(pool), str(i), str(CH), '19120001', '14', str(tmp)], check=True)
    tmp.rename(out); print('done', arm, i, flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex: list(ex.map(run, jobs))
for arm in pr['arms']:
    rows = sorted((json.loads(l) for f in parts.glob(f'{arm}-*.jsonl') for l in f.read_text().splitlines()), key=lambda r: r['index'])
    assert [r['index'] for r in rows] == list(range(N)), arm
    (root/f'cont-{arm}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
print('complete', flush=True)
