"""V44: shapes of 6144+3072+1536+768 in the 970 V40-S games that reach the final-768 chain cards
(6-ply / 0.005 from the first 6144, W31). Descriptive -> results.json."""
import json, itertools
from pathlib import Path
L = Path('training/v44-shape')
R = {r['index']: r for f in (L / 'games').glob('r-*.jsonl') for r in map(json.loads, f.read_text().splitlines())}
V = {r['index']: r for f in Path('training/v40-late-depth/games').glob('S-*.jsonl') for r in map(json.loads, f.read_text().splitlines())}
assert len(R) == 970 and all(all(R[i][k] == V[i][k] for k in ('success', 'moves', 'scoreGained', 'm768')) for i in R), 'replay differs from V40'
snaps = [r for f in (L / 'games').glob('s-*.jsonl') for r in map(json.loads, f.read_text().splitlines())]
adj = lambda a, b: abs(a // 4 - b // 4) + abs(a % 4 - b % 4) == 1
def shape(b):
    pos = {r: [i for i, v in enumerate(b) if v == r] for r in (14, 13, 12, 11)}
    best = None
    for p in itertools.product(*(pos[r] for r in (14, 13, 12, 11))):
        cells = set(p); rows = {c // 4 for c in p}; cols = {c % 4 for c in p}
        chain = adj(p[0], p[1]) and adj(p[1], p[2]) and adj(p[2], p[3])
        square = len(rows) == 2 and len(cols) == 2 and len(cells) == 4
        line = len(rows) == 1 or len(cols) == 1
        conn = len(cells) == 4 and all(any(adj(c, d) for d in cells if d != c) for c in cells)
        conn = conn and len({frozenset(x) for x in [cells]}) == 1 and (chain or square or _connected(cells))
        if square and chain: k = (0, '2x2 square, in order')
        elif line and chain: k = (1, 'straight line')
        elif chain: k = (2, 'L / snake chain')
        elif square: k = (3, '2x2 square, out of order')
        elif conn: k = (4, 'touching, not a chain')
        else: k = (5, 'apart')
        corner = p[0] in (0, 3, 12, 15); edge = all(c // 4 in (0, 3) or c % 4 in (0, 3) for c in p)
        cand = (k, corner, edge)
        if best is None or cand[0][0] < best[0][0]: best = cand
    return {'shape': best[0][1], 'corner6144': best[1], 'allOnEdge': best[2]}
def _connected(cells):
    cells = list(cells); seen = {cells[0]}; todo = [cells[0]]
    while todo:
        c = todo.pop()
        for d in cells:
            if d not in seen and adj(c, d): seen.add(d); todo.append(d)
    return len(seen) == len(cells)
first = {s['index']: s for s in snaps if s['sinceM768'] == 0}
assert len(first) == 970
def table(rows):
    out = {}
    for s in rows:
        k = shape(s['board'])['shape']; o = out.setdefault(k, {'n': 0, 'success': 0}); o['n'] += 1; o['success'] += R[s['index']]['success']
    for o in out.values(): o['P12288'] = o['success'] / o['n']
    tot = sum(o['n'] for o in out.values())
    for o in out.values(): o['share'] = o['n'] / tot
    return dict(sorted(out.items(), key=lambda kv: -kv[1]['n']))
by_game = {}
for s in snaps: by_game.setdefault(s['index'], []).append(s)
ever = {}
for i, ss in by_game.items():
    kinds = {shape(s['board'])['shape'] for s in ss}
    for k in kinds: ever.setdefault(k, [0, 0]); ever[k][0] += 1; ever[k][1] += R[i]['success']
fs = [shape(s['board']) for s in first.values()]
res = {'games': 970, 'replayIdenticalToV40S': True, 'snapshots': len(snaps),
       'atFirstFinal768': table(first.values()),
       'allSnapshotsEvery5Moves': table(snaps),
       'gamesEverInShape': {k: {'games': v[0], 'P12288': v[1] / v[0]} for k, v in sorted(ever.items(), key=lambda kv: -kv[1][0])},
       'atFirstFinal768Corner6144': sum(f['corner6144'] for f in fs) / len(fs),
       'overallP12288': sum(R[i]['success'] for i in R) / 970}
(L / 'results.json').write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))
