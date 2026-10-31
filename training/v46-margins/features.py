"""V46 feature analysis -> features.json: what positions look like when the AI's choice is clear-cut,
a two- or three-way near tie, or when it strongly avoids a direction; split by the number of legal moves.

Gap of a direction = (best root value - its root value) / |best root value| (same measure as the margin table).
Classes, for decisions with k legal directions (gaps sorted g1=0 <= g2 <= g3 <= g4):
  forced  k = 1
  clear   g2 >= 2%                                  one direction is clearly best
  tie2    g2 < 0.25% and (k = 2 or g3 >= 1%)        two directions about equal, the rest clearly worse
  tie3    k >= 3, g3 < 0.25% and (k = 3 or g4 >= 1%)
  tie4    k = 4, g4 < 0.25%
  mixed   everything else
Avoided direction: a legal direction with gap >= 5% (counted per direction, any class).
Directions: 0 left, 1 right, 2 up, 3 down (training/ntuple.h move()).
"""
import json, random
from pathlib import Path
from collections import Counter, defaultdict

L = Path('training/v46-margins')
CLASSES = ['forced', 'clear', 'tie2', 'tie3', 'tie4', 'mixed']
PHASES = [('≤192', 0, 9), ('384–768', 10, 11), ('1536', 12, 12), ('3072', 13, 13), ('6144+', 14, 99)]
CORNERS = {0, 3, 12, 15}
TOWARD = {0: {0, 2}, 3: {1, 2}, 12: {0, 3}, 15: {1, 3}}  # directions that push toward the corner holding the largest card

def phase(r): return next(i for i, (_, lo, hi) in enumerate(PHASES) if lo <= r <= hi)
def mergeable(a, b): return a and b and (a + b == 3 or (a >= 3 and a == b))
def move(b, d):
    out = list(b); lanes = 0
    for lane in range(4):
        idx = [lane * 4 + j if d == 0 else lane * 4 + 3 - j if d == 1 else j * 4 + lane if d == 2 else (3 - j) * 4 + lane for j in range(4)]
        a = [b[i] for i in idx]; ch = False
        for j in range(1, 4):
            if a[j] and (not a[j - 1] or mergeable(a[j], a[j - 1])):
                x, y = a[j], a[j - 1]; a[j - 1] = x if not y else 3 if x + y == 3 else x + 1; a[j] = 0; ch = True
        if ch:
            lanes += 1
            for j in range(4): out[idx[j]] = a[j]
    return out, lanes

def mergemax(b, d):
    """Largest card created by a merge when moving b in direction d (0 = no merge)."""
    top = 0
    for lane in range(4):
        idx = [lane * 4 + j if d == 0 else lane * 4 + 3 - j if d == 1 else j * 4 + lane if d == 2 else (3 - j) * 4 + lane for j in range(4)]
        a = [b[i] for i in idx]
        for j in range(1, 4):
            if a[j] and (not a[j - 1] or mergeable(a[j], a[j - 1])):
                c = a[j] if not a[j - 1] else 3 if a[j] + a[j - 1] == 3 else a[j] + 1
                if a[j - 1]: top = max(top, c)
                a[j - 1] = c; a[j] = 0
    return top

def nbrs(i):
    r, c = divmod(i, 4)
    return [rr * 4 + cc for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)) if 0 <= rr < 4 and 0 <= cc < 4]
def chain(b):
    """Length of the descending run from the largest card: each step to a neighbour one rank lower (ranks >= 3)."""
    i = max(range(16), key=lambda k: b[k]); n = 1; seen = {i}
    while True:
        nxt = [j for j in nbrs(i) if j not in seen and b[j] >= 3 and b[j] == b[i] - 1]
        if not nxt: return n
        i = nxt[0]; seen.add(i); n += 1
def pairs(b): return sum(mergeable(b[i], b[j]) for i in range(16) for j in nbrs(i) if j > i)
def maxcell(b): return max(range(16), key=lambda k: (b[k], -k))
def where(i): return 'corner' if i in CORNERS else 'edge' if i // 4 in (0, 3) or i % 4 in (0, 3) else 'middle'

def classify(g, k):
    if k == 1: return 'forced'
    if g[1] >= 0.02: return 'clear'
    if g[1] < 0.0025 and (k == 2 or g[2] >= 0.01): return 'tie2'
    if k >= 3 and g[2] < 0.0025 and (k == 3 or g[3] >= 0.01): return 'tie3'
    if k == 4 and g[3] < 0.0025: return 'tie4'
    return 'mixed'

def reasons(m, d, best, b):
    """Why direction d scores below the best direction: simple, checkable tags (several can apply)."""
    D, B = m['dirs'][d], m['dirs'][best]; t = []; mc = maxcell(b)
    if D['pDeath'] > B['pDeath'] + 1e-9: t.append('death')
    if D['big'] > B['big']: t.append('bigMoved')
    if mc in CORNERS and d not in TOWARD[mc] and best in TOWARD[mc]: t.append('away')
    if mc in CORNERS and maxcell(move(b, d)[0]) not in CORNERS and maxcell(move(b, best)[0]) in CORNERS: t.append('cornerLeft')
    mb, md = mergemax(b, best), mergemax(b, d)
    if mb >= 8 and mb > md: t.append('missesBigMerge')
    if D['lanes'] < B['lanes'] and D['merges'] < B['merges']: t.append('fewerMerges')
    return t or ['other']

def summary(rows):
    n = len(rows)
    if not n: return None
    avg = lambda f: sum(f(r) for r in rows) / n
    return {'n': n, 'empty': avg(lambda r: r['empty']), 'ones': avg(lambda r: r['ones']), 'twos': avg(lambda r: r['twos']),
            'imbalance': avg(lambda r: abs(r['ones'] - r['twos'])), 'pairs': avg(lambda r: r['pairs']), 'chain': avg(lambda r: r['chain']),
            'maxCorner': avg(lambda r: r['where'] == 'corner'), 'maxMiddle': avg(lambda r: r['where'] == 'middle'),
            'next': {k: v / n for k, v in Counter(r['nextKind'] for r in rows).items()},
            'bestPDeath': avg(lambda r: r['bestPDeath']), 'anyDeathRisk': avg(lambda r: r['anyDeath'])}

rows, heat = [], defaultdict(lambda: {'max': [0] * 16, 'empty': [0] * 16, 'n': 0})
examples = defaultdict(list)
avoid = defaultdict(Counter); avoidN = Counter(); runner = defaultdict(Counter); corner = defaultdict(Counter)
rng = random.Random(46); seen = Counter()
for f in sorted(L.glob('data/m-*.jsonl')):
    for line in f.open():
        m = json.loads(line); b = m['board']; legal = [d for d in range(4) if m['dirs'][d]['legal']]
        k = len(legal); vals = {d: m['dirs'][d]['value'] for d in legal}; best = max(legal, key=lambda d: vals[d])
        top = vals[best]; g = sorted((top - vals[d]) / abs(top) if top else 0.0 for d in legal)
        cls = classify(g, k); ph = phase(m['maxRank'])
        nc = m['next']['cards']
        r = {'ph': ph, 'k': k, 'cls': cls, 'empty': b.count(0), 'ones': b.count(1), 'twos': b.count(2), 'pairs': pairs(b), 'chain': chain(b),
             'where': where(maxcell(b)), 'nextKind': 'bonus' if m['next']['bonus'] else {1: '1', 2: '2'}.get(nc[0], '3+'),
             'bestPDeath': m['dirs'][best]['pDeath'], 'anyDeath': any(m['dirs'][d]['pDeath'] > 0 for d in legal)}
        rows.append(r)
        h = heat[(ph >= 3, cls)]; h['n'] += 1; h['max'][maxcell(b)] += 1
        for i in range(16): h['empty'][i] += b[i] == 0
        order = sorted(legal, key=lambda d: -vals[d]); late = ph >= 3
        for d in legal:
            if d != best and top and (top - vals[d]) / abs(top) >= 0.05:
                avoidN[ph] += 1
                for t in reasons(m, d, best, b): avoid[ph][t] += 1
        if cls == 'clear':
            runner[late]['n'] += 1; runner[late]['bonusNext'] += m['next']['bonus']
            for t in reasons(m, order[1], best, b): runner[late][t] += 1
        mc = maxcell(b); c = corner[late]
        if mc in CORNERS and top:
            T = TOWARD[mc]; nt = len(T & set(legal)); c[f'{cls}|n'] += 1; c[f'{cls}|towardLegal{nt}'] += 1; c[f'towardLegal{nt}'] += 1
            if cls == 'tie2': c['tie2|pairToward'] += set(order[:2]) == T; c['tie2|pairOpposite'] += set(order[:2]) in ({0, 1}, {2, 3})
            if cls == 'tie3' and k == 4: c['tie3k4|n'] += 1; c['tie3k4|excludedAway'] += order[3] not in T
            if cls == 'clear': c['clear|bestToward'] += order[0] in T
            for d in order[1:]:
                if (top - vals[d]) / abs(top) >= 0.05: c['avoid|n'] += 1; c['avoid|away'] += d not in T
        else: c[f'{cls}|noCorner'] += 1
        key = (cls, k, ph >= 3); seen[key] += 1; ex = examples[key]
        if len(ex) < 6: ex.append(m)
        else:
            j = rng.randrange(seen[key])
            if j < 6: ex[j] = m

def slim(m):
    legal = [d for d in range(4) if m['dirs'][d]['legal']]; top = max(m['dirs'][d]['value'] for d in legal)
    return {'board': m['board'], 'next': m['next'], 'maxRank': m['maxRank'], 'chosen': m['chosen'],
            'dirs': [None if not x['legal'] else {'gap': (top - x['value']) / abs(top) if top else 0, 'pDeath': x['pDeath'], 'big': x['big'],
                     'merges': x['merges'], 'lanes': x['lanes']} for x in m['dirs']]}

out = {'moves': len(rows), 'phases': [p[0] for p in PHASES], 'classes': CLASSES,
       'byPhase': [{c: sum(1 for r in rows if r['ph'] == p and r['cls'] == c) for c in CLASSES} for p in range(len(PHASES))],
       'byLegal': {k: {c: sum(1 for r in rows if r['k'] == k and r['cls'] == c) for c in CLASSES} for k in (1, 2, 3, 4)},
       'legalByPhase': [{k: sum(1 for r in rows if r['ph'] == p and r['k'] == k) for k in (1, 2, 3, 4)} for p in range(len(PHASES))],
       'features': {f'{late}|{c}': summary([r for r in rows if (r['ph'] >= 3) == late and r['cls'] == c]) for late in (False, True) for c in CLASSES},
       'featuresByLegal': {f'{late}|{k}': summary([r for r in rows if (r['ph'] >= 3) == late and r['k'] == k]) for late in (False, True) for k in (1, 2, 3, 4)},
       'heat': {f'{late}|{c}': heat[(late, c)] for (late, c) in heat},
       'avoid': [{'n': avoidN[p], **avoid[p]} for p in range(len(PHASES))],
       'runnerUp': {str(k): dict(v) for k, v in runner.items()}, 'corner': {str(k): dict(v) for k, v in corner.items()},
       'examples': {f'{c}|{k}|{late}': [slim(m) for m in examples[(c, k, late)]] for (c, k, late) in examples}}
(L / 'features.json').write_text(json.dumps(out, ensure_ascii=False, separators=(',', ':')) + '\n')
print(json.dumps({k: out[k] for k in ('moves', 'byPhase', 'byLegal', 'legalByPhase', 'avoid', 'runnerUp', 'corner')}, ensure_ascii=False))
for k, v in out['features'].items(): print(k, v and {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
for k, v in out['featuresByLegal'].items(): print('legal', k, v and {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a in ('n', 'empty', 'imbalance', 'pairs', 'maxCorner', 'anyDeathRisk')})
