"""V24 runner (from repo root). Build: g++ -std=c++17 -O3 -march=native training/v24-acceptance/game.cpp -o /tmp/v24-game"""
from pathlib import Path
import subprocess, json, concurrent.futures as cf
R = Path('training/v24-acceptance'); pr = json.loads((R/'protocol.json').read_text()); out = R/'games'; out.mkdir(exist_ok=True)
ARGS = {'OLD': ['dist/models/ntuple-v4.bin','3','0','24000','0'], 'NEW': ['dist/models/ntuple-v23.bin','5','0.01','0','1']}
first, last = pr['seeds']; CH = 32
jobs = [(a, s) for s in range(first, last+1, CH) for a in ('NEW', 'OLD')]
def run(j):
    a, s = j; f = out/f'{a}-{s}.jsonl'
    if f.exists() and len(f.read_text().splitlines()) == CH: return
    tmp = out/f'{a}-{s}.part'; tmp.unlink(missing_ok=True)
    subprocess.run(['/tmp/v24-game', *ARGS[a], str(s), str(CH), str(tmp)], check=True); tmp.rename(f); print('done', a, s, flush=True)
with cf.ThreadPoolExecutor(2) as ex: list(ex.map(run, jobs))
for a in ARGS:
    rows = sorted((json.loads(l) for f in out.glob(f'{a}-*.jsonl') for l in f.read_text().splitlines()), key=lambda r: r['seed'])
    assert [r['seed'] for r in rows] == list(range(first, last+1)), a
    (R/f'games-{a}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
print('complete', flush=True)
