"""Fixed V18 protocol runner, resumable. Build: g++ -std=c++17 -O3 -march=native training/v18-late/game.cpp -o /tmp/v18-game"""
from pathlib import Path
import concurrent.futures, subprocess, json, hashlib, sys
root = Path('training/v18-late'); pr = json.loads((root/'protocol.json').read_text())
assert hashlib.sha256(Path('training/v7/base.ntd').read_bytes()).hexdigest() == pr['baseSHA256']
first, last = pr['seeds']; CH = 32; parts = root/'games'; parts.mkdir(exist_ok=True)
workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
jobs = [(a, s) for s in range(first, last+1, CH) for a in pr['arms']]   # interleaved by seed chunk
def run(job):
    arm, s = job; cfg = pr['arms'][arm]; out = parts/f'{arm}-{s}.jsonl'
    if out.exists() and len(out.read_text().splitlines()) == CH: return
    tmp = out.with_suffix('.tmp'); tmp.unlink(missing_ok=True)
    subprocess.run(['/tmp/v18-game', str(cfg['depth']), str(cfg['threshold']), str(s), str(CH), str(tmp)], check=True)
    tmp.rename(out); print('done', arm, s, flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex: list(ex.map(run, jobs))
for arm in pr['arms']:
    rows = sorted((json.loads(l) for f in parts.glob(f'{arm}-*.jsonl') for l in f.read_text().splitlines()), key=lambda r: r['seed'])
    assert [r['seed'] for r in rows] == list(range(first, last+1)), arm
    (root/f'games-{arm}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
print('complete', flush=True)
