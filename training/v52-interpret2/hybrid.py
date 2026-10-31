"""V52 question 2b: does the model's own confidence tell when to ask the AI?
300 whole games (seeds 50900001.., as V51) with the DAgger-14 transformer; on a move where the softmax
probability of its choice is below a threshold, the website AI's 6-ply search (bin/label, 12 processes)
chooses instead. Thresholds 0.447 / 0.508 / 0.554 = the least confident 10% / 20% / 30% of the validation
positions (errors.json). Control: the search on a random 20% of moves.
  uv run --project training/v50-transformer python training/v52-interpret2/hybrid.py  -> hybrid.json"""
import json, random, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
L = Path(__file__).resolve().parent; sys.path.insert(0, str(L.parent / 'v51-interpret'))
from common import load_net, to_dev, run, V50
from engine import Game, move
Q = L.parent / 'web-release' / 'models'
CMD = [str(V50 / 'bin' / 'label'), str(Q / 'ntuple-v31.bin'), ','.join(str(Q / f) for f in ('v31-s2q.res', 'v31-s1q.res', 'v31-s3q.res')), '-', '/dev/stdout']
workers = [subprocess.Popen(CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1) for _ in range(12)]
def search(games):
    for k, g in enumerate(games):
        workers[k % 12].stdin.write(' '.join(map(str, [*g.b, len(g.next['cards']), *g.next['cards'], int(g.next['bonus']), *g.counts, *g.cycle])) + '\n')
    for w in workers: w.stdin.flush()
    return [int(np.nanargmax([float(x) for x in workers[k % 12].stdout.readline().split()])) for k in range(len(games))]

net = load_net()
def play(rule, n=300):
    games = [Game(50900001 + i) for i in range(n)]; alive = list(games); asked = total = 0; rng = random.Random(52); t0 = time.time()
    while alive:
        x = {'board': [g.b for g in alive], 'cards': [(g.next['cards'] + [0, 0, 0])[:3] for g in alive], 'bonus': [g.next['bonus'] for g in alive],
             'counts': [g.counts for g in alive], 'cycle': [list(g.cycle) for g in alive], 'legal': [[bool(move(g.b, d)[1]) for d in range(4)] for g in alive]}
        with torch.no_grad(): p = torch.softmax(run(net, to_dev({k: np.array(v) for k, v in x.items()}))[0], 1).cpu().numpy()
        ds = p.argmax(1).tolist()
        ask = [k for k in range(len(alive)) if (rule[0] == 'p' and p[k].max() < rule[1]) or (rule[0] == 'random' and rng.random() < rule[1])]
        if ask:
            for k, d in zip(ask, search([alive[k] for k in ask])): ds[k] = d
        asked += len(ask); total += len(alive)
        for g, d in zip(alive, ds): g.step(d)
        alive = [g for g in alive if not g.over and g.turns < 50000]
    top = [max(g.b) for g in games]
    return {'asked': asked / total, 'meanScore': float(np.mean([g.score() for g in games])), 'meanMoves': float(np.mean([g.turns for g in games])),
            **{f'reach{c}': float(np.mean([t >= r for t in top])) for c, r in (('1536', 12), ('3072', 13), ('6144', 14), ('12288', 15))}, 'sec': round(time.time() - t0)}
out = {}
for name, rule in [('只用模型', ('p', 0)), ('最没把握的约 10%', ('p', 0.447)), ('最没把握的约 20%', ('p', 0.508)), ('最没把握的约 30%', ('p', 0.554)), ('随机 20%', ('random', 0.2))]:
    out[name] = play(rule); print(name, {k: round(v, 4) for k, v in out[name].items()}, flush=True)
    (L / 'hybrid.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
for w in workers: w.stdin.close(); w.wait()
