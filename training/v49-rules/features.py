"""V49 step 1: simple, human-checkable features of each legal direction -> cache.npz.
Positions: the 200 AI games of V46 (data/m-*.jsonl, website AI with cycle) and the human games of
V47 (video) and V48 (LogicCube), each with the website AI's root value for every legal direction.
Boards in ranks: 1, 2, 3, then 6 = 4, 12 = 5, ... 6144 = 14. Directions 0 left, 1 right, 2 up, 3 down.
Every feature is computed from the board right after the move (before the new card enters),
except `toward` (direction only) and `safe` / `fit`, which also look at the preview card;
`setup` / `setupBig` look one move further (largest merge available next move, ignoring the new
card; number of next directions that would merge a card of 24 or more).
  python features.py  (about a minute on 16 processes)"""
import json, math
import numpy as np
from pathlib import Path
from multiprocessing import Pool
L = Path(__file__).resolve().parent; T = L.parent
CORNERS = (0, 3, 12, 15); TOWARD = {0: (0, 2), 3: (1, 2), 12: (0, 3), 15: (1, 3)}
FEATURES = ['bigStill', 'toward', 'maxCorner', 'lineStill', 'merges', 'mergeMax', 'lanes', 'pairs', 'chain', 'secondAdj',
            'smooth', 'mono', 'edgeSum', 'safe', 'fit', 'balance', 'movedMax', 'movedBig', 'trapped', 'empties', 'setup', 'setupBig']
LANES = [[[r * 4 + j for j in range(4)] for r in range(4)], [[r * 4 + 3 - j for j in range(4)] for r in range(4)],
         [[j * 4 + c for j in range(4)] for c in range(4)], [[(3 - j) * 4 + c for j in range(4)] for c in range(4)]]
NB = [[j for j in (i - 4, i + 4, i - 1 if i % 4 else -1, i + 1 if i % 4 < 3 else -1) if 0 <= j < 16] for i in range(16)]
ADJ = [(i, j) for i in range(16) for j in NB[i] if j > i]
LINES = [[r * 4 + c for c in range(4)] for r in range(4)] + [[r * 4 + c for r in range(4)] for c in range(4)]

def mergeable(a, b): return a and b and (a + b == 3 if a < 3 or b < 3 else a == b)
def rank(v): return v if v <= 3 else round(math.log2(v / 3)) + 3

def move(b, d):
    out = list(b); moved = set(); entries = []; merges = 0; top = 0
    for idx in LANES[d]:
        a = [b[i] for i in idx]
        j0 = next((j for j in range(1, 4) if a[j] and (not a[j - 1] or mergeable(a[j], a[j - 1]))), None)
        if j0 is None: continue
        if a[j0 - 1]:
            c = 3 if a[j0] + a[j0 - 1] == 3 else a[j0] + 1; merges += 1; top = max(top, c)
        else: c = a[j0]
        a[j0 - 1] = c; a[j0:] = a[j0 + 1:] + [0]
        for j in range(j0, 4):
            if b[idx[j]]: moved.add(idx[j])
        for j in range(4): out[idx[j]] = a[j]
        entries.append((idx[3], idx[2]))
    return out, moved, entries, merges, top

def maxcell(b): return max(range(16), key=lambda k: (b[k], -k))
def chain(b):
    i = maxcell(b); n = 1; seen = {i}
    while True:
        nxt = [j for j in NB[i] if j not in seen and b[j] >= 3 and b[j] == b[i] - 1]
        if not nxt: return n
        i = nxt[0]; seen.add(i); n += 1
def dead(b): return all(b) and not any(mergeable(b[i], b[j]) for i, j in ADJ)

def features(b, cards):
    """[4, F] feature values (nan for an illegal direction)."""
    out = np.full((4, len(FEATURES)), np.nan); mc = maxcell(b)
    for d in range(4):
        a, moved, entries, merges, top = move(b, d)
        if not entries: continue
        m2 = maxcell(a); rest = sorted(a, reverse=True)
        second = rest[1]
        safe = not any(dead(a[:e] + [c] + a[e + 1:]) for e, _ in entries for c in cards)
        fit = sum(any(mergeable(c, a[n]) for c in cards) for _, n in entries) / len(entries)
        out[d] = [mc not in moved, d in TOWARD.get(mc, ()), m2 in CORNERS,
                  sum(all(a[i] == b[i] for i in line) for line in LINES if mc in line),
                  merges, top, len(entries),
                  sum(bool(mergeable(a[i], a[j])) for i, j in ADJ), chain(a),
                  any(a[j] == second for j in NB[m2]),
                  -sum(abs(a[i] - a[j]) for i, j in ADJ if a[i] >= 3 and a[j] >= 3),
                  sum(all(x >= y for x, y in zip(l, l[1:])) or all(x <= y for x, y in zip(l, l[1:])) for l in ([a[i] for i in line] for line in LINES)),
                  max(sum(a[i] for i in line) for line in (LINES[0], LINES[3], LINES[4], LINES[7])),
                  safe, fit, -abs(a.count(1) - a.count(2)),
                  max((b[i] for i in moved), default=0), sum(b[i] >= 6 for i in moved),
                  sum(1 <= a[i] <= 3 and not any(a[j] == 0 or mergeable(a[i], a[j]) for j in NB[i]) for i in range(16)),
                  a.count(0),
                  max(move(a, d2)[4] for d2 in range(4)), sum(move(a, d2)[4] >= 6 for d2 in range(4))]
    return out

def load():
    """(source, game, board ranks, preview card ranks, values[4] with nan for illegal)."""
    for f in sorted((T / 'v46-margins' / 'data').glob('m-*.jsonl')):
        for line in f.open():
            m = json.loads(line)
            yield 'ai', m['seed'], m['board'], m['next']['cards'], [x['value'] if x['legal'] else math.nan for x in m['dirs']]
    sources = [('v47', T / 'v47-human' / 'ai.jsonl')] + [('v48', p) for p in sorted((T / 'v48-logiccube' / 'ai').glob('*.jsonl'))]
    for src, p in sources:
        for line in p.open():
            r = json.loads(line); nx = r['next']
            cards = sorted({rank(c) for w in nx.get('windows', [nx.get('candidates')]) for c in w})
            yield src, int(p.stem) if src == 'v48' else 0, [rank(v) for row in r['board'] for v in row], cards, [math.nan if v is None else v for v in r['values']]

def work(x): return features(x[2], x[3])

if __name__ == '__main__':
    rows = list(load())
    with Pool(16) as pool: X = np.stack(pool.map(work, rows, chunksize=2000))
    V = np.array([r[4] for r in rows], dtype=float)
    legal = ~np.isnan(X[:, :, 0])
    assert (legal == ~np.isnan(V)).all(), 'legal directions disagree with the search'
    np.savez_compressed(L / 'cache.npz', X=X.astype(np.float32), V=V, src=np.array([r[0] for r in rows]), game=np.array([r[1] for r in rows]),
                        maxRank=np.array([max(r[2]) for r in rows]), features=np.array(FEATURES))
    print(len(rows), 'positions', {s: sum(r[0] == s for r in rows) for s in ('ai', 'v47', 'v48')})
