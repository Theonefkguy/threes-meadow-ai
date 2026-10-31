"""V49 step 2: a short ordered rule list that imitates the website AI -> rules.json.
A rule = one feature and a sign ("prefer more X" / "prefer less X"), or a threshold on it
("prefer X >= t" / "prefer X <= t"), which keeps every direction that passes when any does. The list is applied in order:
keep only the legal directions with the best value of rule 1, then of rule 2 among those, and so on;
whatever is still tied is chosen uniformly at random (metrics are the expectation over that choice).
Rules are added greedily, each time the one that most raises the share of moves the AI rates within
0.25% of its best ("near"), measured on the even-seed V46 games (train). Reported on the odd-seed
games (test) and on the human positions of V47 / V48 next to what the humans actually played.
Forced positions (one legal direction) are left out.
  uv run --no-project --python 3.12 --with numpy python fit.py"""
import json
import numpy as np
from pathlib import Path
L = Path(__file__).resolve().parent
z = np.load(L / 'cache.npz'); X, V, src, game, maxRank, FEAT = z['X'], z['V'], z['src'], z['game'], z['maxRank'], list(z['features'])
legal = ~np.isnan(V); keep = legal.sum(1) >= 2
X, V, src, game, maxRank, legal = X[keep], V[keep], src[keep], game[keep], maxRank[keep], legal[keep]
top = np.nanmax(V, 1); gap = (top[:, None] - V) / np.abs(top)[:, None]; gap[~legal] = np.nan
best = np.nanargmax(V, 1)
NEAR, BAD = gap < 0.0025, gap >= 0.05
train = (src == 'ai') & (game % 2 == 0); test = (src == 'ai') & (game % 2 == 1)
PHASES = [('≤192', 0, 9), ('384–768', 10, 11), ('1536', 12, 12), ('3072', 13, 13), ('6144+', 14, 99)]
MAXRULES = 10

def apply(rules, rows=slice(None)):
    k = legal[rows].copy()
    for f, s, t in rules:
        v = s * X[rows][:, :, f]
        if t is not None: v = (v >= s * t).astype(float)
        v = np.where(k, v, -np.inf); k &= v == v.max(1, keepdims=True)
    return k
def metrics(k, rows):
    w = k / k.sum(1, keepdims=True); n = len(w)
    return {'n': int(n), 'near': float((w * NEAR[rows]).sum() / n), 'match': float(w[np.arange(n), best[rows]].sum() / n),
            'bad5': float((w * BAD[rows]).sum() / n), 'meanGap': float((w * np.nan_to_num(gap[rows])).sum() / n), 'tied': float((k.sum(1) > 1).mean())}

GROUPS = {'all': (0, 99), 'early': (0, 11), 'mid': (12, 12), 'late': (13, 99)}   # by the biggest card: ≤768, 1536, ≥3072
def phase_rows(rows, lo, hi): return rows & (maxRank >= lo) & (maxRank <= hi)

def greedy(idx, label):
    rules, path = [], []
    for step in range(MAXRULES):
        base = metrics(apply(rules, idx), idx)['near']; cand = []
        for f in range(len(FEAT)):
            if any(f == g and t is None for g, _, t in rules): continue
            vals = np.unique(X[idx][:, :, f][legal[idx]])
            ts = [None] + list(vals[1:] if len(vals) <= 12 else np.unique(np.quantile(X[idx][:, :, f][legal[idx]], np.linspace(.05, .95, 12))))
            for s in (1, -1):
                for t in ts:
                    r = (f, s, t)
                    if r in rules: continue
                    cand.append((metrics(apply(rules + [r], idx), idx)['near'], f, s, t))
        gain, f, s, t = max(cand, key=lambda c: c[0])
        if gain - base < 0.001: break
        rules.append((f, s, t))
        name = ('more ' if s > 0 else 'less ') + FEAT[f] if t is None else f"{FEAT[f]} {'>=' if s > 0 else '<='} {t:g}"
        path.append({'rule': name, 'feature': FEAT[f], 'sign': s, 'threshold': None if t is None else float(t), 'trainNear': gain})
        print(f"{label:5s} {step + 1:2d} {name:20s} train near {gain:.1%} (+{gain - base:.1%})")
    return rules, path

fitted = {g: greedy(np.flatnonzero(phase_rows(train, lo, hi)), g) for g, (lo, hi) in GROUPS.items()}
def policy(kind, n, rows):
    """kind 'all': one list for every phase; 'phase': the early / mid / late list by the biggest card. First n rules."""
    k = np.zeros_like(legal[rows])
    for g, (lo, hi) in (GROUPS.items() if kind == 'phase' else [('all', GROUPS['all'])]):
        if g == 'all' and kind == 'phase': continue
        m = (maxRank[rows] >= lo) & (maxRank[rows] <= hi)
        if m.any(): k[m] = apply(fitted[g][0][:n], np.flatnonzero(rows)[m])
    return k
def report(rows):
    out = {}
    for kind in ('all', 'phase'):
        out[kind] = []
        for n in range(MAXRULES + 1):
            if n > max(len(fitted[g][0]) for g in (['all'] if kind == 'all' else ['early', 'mid', 'late'])): break
            k = policy(kind, n, rows); r = metrics(k, np.flatnonzero(rows))
            r['byPhase'] = {name: metrics(k[(maxRank[rows] >= lo) & (maxRank[rows] <= hi)], np.flatnonzero(phase_rows(rows, lo, hi)))
                            for name, lo, hi in PHASES if phase_rows(rows, lo, hi).any()}
            out[kind].append(r)
    return out
result = {'features': FEAT, 'rules': {g: fitted[g][1] for g in GROUPS}, 'test': report(test)}
for s in ('v47', 'v48'):
    rows = src == s; result[s] = report(rows)
    # what the human actually played, from the source analyses
    a = json.loads((L.parent / ('v47-human' if s == 'v47' else 'v48-logiccube') / 'analysis.json').read_text())
    o = a if s == 'v47' else a['pooled']
    pick = lambda x: {'n': x['n'], 'near': x['within025'], 'match': x['agree'], 'bad5': x['over5'], 'meanGap': x['meanGap']}
    result[s + 'Human'] = {**pick(o['overall']), 'byPhase': {k: pick(v) for k, v in o['byPhase'].items() if v}}
(L / 'rules.json').write_text(json.dumps(result, ensure_ascii=False, indent=1) + '\n')
for kind in ('all', 'phase'):
    print('==', kind, ' rules  test(AI games)  |  V47 video  |  V48 LogicCube')
    for n, (t, a, b) in enumerate(zip(result['test'][kind], result['v47'][kind], result['v48'][kind])):
        print(f"{n:2d}  near {t['near']:.1%} match {t['match']:.1%} bad5 {t['bad5']:.1%} | near {a['near']:.1%} bad5 {a['bad5']:.1%} | near {b['near']:.1%} bad5 {b['bad5']:.1%}")
        if n == len(result['test'][kind]) - 1:
            for name in result['v48'][kind][n]['byPhase']:
                x, h = result['v48'][kind][n]['byPhase'][name], result['test'][kind][n]['byPhase'][name]
                print(f"     {name:8s} test near {h['near']:.1%} bad5 {h['bad5']:.1%} | v48 near {x['near']:.1%} bad5 {x['bad5']:.1%}")
for s in ('v47', 'v48'): print(s, 'human', {k: (round(v, 3) if not isinstance(v, dict) else {p: (round(q['near'], 3), round(q['bad5'], 3)) for p, q in v.items()}) for k, v in result[s + 'Human'].items()})
