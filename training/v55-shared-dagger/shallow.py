"""V55 reference: the website AI's n-tuple evaluation with a shallow search (depth 1, 2, 3 instead of 6),
whole games on the evaluation seeds (500 games from 50900001, as eval.py), 12 bin/label processes.
  uv run --project ../v50-transformer python shallow.py [DEPTH ...]   -> results/shallow.json"""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
L = Path(__file__).resolve().parent; V50 = L.parent / 'v50-transformer'; sys.path.insert(0, str(V50))
from engine import Game
Q = L.parent / 'web-release' / 'models'
CMD = [str(V50 / 'bin' / 'label'), str(Q / 'ntuple-v31.bin'), ','.join(str(Q / f) for f in ('v31-s2q.res', 'v31-s1q.res', 'v31-s3q.res')), '-', '/dev/stdout']

def play(depth, n=500):
    ws = [subprocess.Popen(CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env={**os.environ, 'LABEL_DEPTH': str(depth)}) for _ in range(12)]
    games = [Game(50900001 + i) for i in range(n)]; alive = list(games); t0 = time.time()
    while alive:
        for k, g in enumerate(alive):
            ws[k % 12].stdin.write(' '.join(map(str, [*g.b, len(g.next['cards']), *g.next['cards'], int(g.next['bonus']), *g.counts, *g.cycle])) + '\n')
        for w in ws: w.stdin.flush()
        ds = [int(np.nanargmax([float(x) for x in ws[k % 12].stdout.readline().split()])) for k in range(len(alive))]
        for g, d in zip(alive, ds): g.step(d)
        alive = [g for g in alive if not g.over and g.turns < 50000]
    for w in ws: w.stdin.close(); w.wait()
    top = [max(g.b) for g in games]
    return {'meanScore': float(np.mean([g.score() for g in games])), 'meanMoves': float(np.mean([g.turns for g in games])),
            **{f'reach{c}': float(np.mean([t >= k for t in top])) for c, k in (('1536', 12), ('3072', 13), ('6144', 14), ('12288', 15))}, 'sec': round(time.time() - t0)}

f = L / 'results' / 'shallow.json'; out = json.loads(f.read_text()) if f.exists() else {}
for d in (sys.argv[1:] or ['1', '2']):
    out[f'depth{d}'] = play(int(d)); print(d, out[f'depth{d}'], flush=True); f.write_text(json.dumps(out, indent=1) + '\n')
