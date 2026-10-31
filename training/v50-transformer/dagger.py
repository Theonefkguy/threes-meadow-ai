"""V50 DAgger step: the student (a transformer) plays whole games with no search; every position it
reaches is labelled with the website AI's root values (bin/label, the V46 search) for the next
round of training. Seeds 51000001 + 10000 * ROUND (not used anywhere else).
LATE > 0 adds LATE late starts (aistart.py): half where the AI first had 1536, half where it first had
3072, in training AI games drawn at random per round; the student plays on from there.
  uv run python dagger.py ROUND MODEL GAMES [PROCS] [LATE]  ->  $DATA/dagger/round-ROUND.npz"""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np
from engine import Game
from play import transformer
from aistart import late_games, ai_seeds
L = Path(__file__).resolve().parent; DATA = Path('/Volumes/Extreme SSD/Datasets/threes-v50'); Q = L.parent / 'web-release' / 'models'

if __name__ == '__main__':
    rnd, model, n = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]); procs = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    late = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    out = DATA / 'dagger' / f'round-{rnd}.npz'; work = DATA / 'dagger' / f'round-{rnd}'; work.mkdir(parents=True, exist_ok=True)
    if out.exists(): print('exists', out); sys.exit(0)
    choose = transformer(model); seed0 = 51000001 + 10000 * rnd
    games = [Game(seed0 + i) for i in range(n)]; states = []; t0 = time.time(); start = {id(g): 0 for g in games}
    pick = np.random.default_rng(rnd).permutation(ai_seeds(False))
    for k, r in enumerate((12, 13)):
        lg = [g for g, _ in late_games(pick[k * late:(k + 1) * late], r)][:late // 2]
        start.update({id(g): r for g in lg}); games += lg
    alive = list(games)
    while alive:
        for g in alive: states.append((g.b[:], g.next['cards'][:], g.next['bonus'], g.counts[:], list(g.cycle), start[id(g)], g.turns))
        for g, d in zip(alive, choose(alive)): g.step(d)
        alive = [g for g in alive if not g.over and g.turns < 50000]
    lt = {r: [max(g.b) for g in games if start[id(g)] == r] for r in (12, 13)}
    if late: print('late starts: from 1536', len(lt[12]), f'reach 3072 {np.mean([t >= 13 for t in lt[12]]):.1%};',
                   'from 3072', len(lt[13]), f'reach 6144 {np.mean([t >= 14 for t in lt[13]]):.1%}', flush=True)
    games = games[:n]; top = [max(g.b) for g in games]
    print(f'round {rnd}: {n} games, {len(states)} positions, mean moves {sum(g.turns for g in games) / n:.0f}, '
          f'reach 768 {np.mean([t >= 11 for t in top]):.1%} 1536 {np.mean([t >= 12 for t in top]):.1%} 3072 {np.mean([t >= 13 for t in top]):.1%} ({time.time() - t0:.0f} s)', flush=True)
    parts = [states[i::procs] for i in range(procs)]; jobs = []
    for i, part in enumerate(parts):
        (work / f'{i}.in').write_text(''.join(' '.join(map(str, [*b, len(c), *c, int(bo), *cn, *cy])) + '\n' for b, c, bo, cn, cy, _, _ in part))
        jobs.append(subprocess.Popen([str(L / 'bin' / 'label'), str(Q / 'ntuple-v31.bin'), ','.join(str(Q / f) for f in ('v31-s2q.res', 'v31-s1q.res', 'v31-s3q.res')),
                                      str(work / f'{i}.in'), str(work / f'{i}.out')]))
    for j in jobs:
        if j.wait(): raise SystemExit('label failed')
    rows = []
    for i, part in enumerate(parts):
        vals = [[float(x) for x in l.split()] for l in (work / f'{i}.out').read_text().splitlines()]
        assert len(vals) == len(part)
        rows += [(s, v) for s, v in zip(part, vals)]
    m = len(rows)
    d = {'board': np.array([s[0] for s, _ in rows], np.uint8), 'cards': np.array([(s[1] + [0, 0, 0])[:3] for s, _ in rows], np.uint8),
         'bonus': np.array([s[2] for s, _ in rows]), 'counts': np.array([s[3] for s, _ in rows], np.uint8), 'cycle': np.array([s[4] for s, _ in rows], np.uint8),
         'values': np.array([v for _, v in rows], np.float32), 'seed': np.full(m, rnd), 'move': np.array([s[6] for s, _ in rows], np.int32), 'start': np.array([s[5] for s, _ in rows], np.uint8)}
    np.savez(out, **d)
    (DATA / 'dagger' / f'round-{rnd}.json').write_text(json.dumps({'round': rnd, 'model': model, 'games': n, 'positions': m, 'top': top, 'late': {str(r): v for r, v in lt.items()}}) + '\n')
    print('labelled', m, f'({time.time() - t0:.0f} s)')
