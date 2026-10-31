"""V61 step 2 analysis (setup in RL-V61-CARD-LUCK.md): death within 30 moves against the card sequence of each
replay (first 10 cards after the preview; 20 as a secondary), compared within positions: Mantel-Haenszel risk
ratios stratified by position, 95% bootstrap over positions (2000). Also how much of the variance in death lies
between positions.  -> results2.json"""
import json
from pathlib import Path
import numpy as np
L = Path(__file__).resolve().parent
rows = [json.loads(l) for f in sorted((L / 'rolls').glob('r-*.jsonl')) for l in f.open()]
meta = {m['id']: m for m in json.loads((L / 'positions.json').read_text())}
pos = np.array([r['pos'] for r in rows]); dead = np.array([r['dead'] for r in rows], float)
script = np.array([r['script'] for r in rows]); top = np.array([meta[p]['top'] for p in pos])  # by position id
pid = pos; pos = np.unique(pid, return_inverse=True)[1]; P = pos.max() + 1          # dense position index

def feats(c):
    n1, n2 = (c == 1).sum(1), (c == 2).sum(1); run = np.zeros(len(c), int)
    for i, row in enumerate(c):
        side = cur = best = 0
        for x in row:
            if x in (1, 2):
                cur = cur + 1 if x == side else 1; side = x; best = max(best, cur)
        run[i] = best
    return {'nBonus': (c >= 4).sum(1), 'absD12': np.abs(n1 - n2), 'maxRun': run, 'n3': (c == 3).sum(1)}

rng = np.random.default_rng(61); W = rng.multinomial(P, np.ones(P) / P, size=2000).astype(float)
def mh(exposed, ref):
    """Mantel-Haenszel risk ratio of death, exposed vs ref, stratified by position; bootstrap over positions."""
    m = exposed | ref; e = exposed[m]; d = dead[m]; p = pos[m]
    a = np.bincount(p[e], d[e], P); n1 = np.bincount(p[e], minlength=P).astype(float)
    c = np.bincount(p[~e], d[~e], P); n0 = np.bincount(p[~e], minlength=P).astype(float); N = n1 + n0
    with np.errstate(invalid='ignore', divide='ignore'): num = np.where(N > 0, a * n0 / N, 0); den = np.where(N > 0, c * n1 / N, 0)
    bs = (W @ num) / (W @ den)
    return {'rr': float(num.sum() / den.sum()), 'ci95': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            'exposed': int(e.sum()), 'deathExposed': float(d[e].mean()), 'deathRef': float(d[~e].mean())}

out = {'replays': len(rows), 'positions': int(P), 'death30': float(dead.mean())}
pm = np.bincount(pos, dead, P) / np.bincount(pos, minlength=P); n = np.bincount(pos, minlength=P)
within = float(np.mean(pm * (1 - pm) * n / (n - 1))); total = float(dead.var())
out['variance'] = {'total': total, 'withinPosition': within, 'betweenShare': 1 - within / total,
                   'positionDeathRate': {'p10': float(np.percentile(pm, 10)), 'median': float(np.median(pm)), 'p90': float(np.percentile(pm, 90))}}
for K in (10, 20):
    F = feats(script[:, :K]); res = {}
    if K == 10:
        res['primary'] = {'bonus2_vs_0to1': mh(F['nBonus'] >= 2, F['nBonus'] <= 1),
                          'absD12_ge4_vs_le2': mh(F['absD12'] >= 4, F['absD12'] <= 2),
                          'maxRun_ge5_vs_le3': mh(F['maxRun'] >= 5, F['maxRun'] <= 3)}
        big = np.array([any(4 <= x < 15 and t - x <= 4 for x in row) for row, t in zip(script[:, :K], top)])
        small = np.array([any(4 <= x < 15 and t - x > 4 for x in row) for row, t in zip(script[:, :K], top)])
        res['bonusSize'] = {'bigBonus_vs_none': mh(big & (F['nBonus'] == 1), F['nBonus'] == 0),
                            'smallBonus_vs_none': mh(small & (F['nBonus'] == 1), F['nBonus'] == 0)}
    # nBonus 2 vs 0 is not estimable within positions: every position where 2 bonuses can come in 10 cards has at least 1
    levels = {'nBonus': (0, [1, 2]), 'absD12': (1, [0, 2, 3, 4, 5, 6]), 'maxRun': (2, [1, 3, 4, 5, 6, 7]),
              'n3': (3, [0, 1, 2, 4, 5, 6]) if K == 10 else (6, [4, 5, 7, 8, 9])}
    res['levels'] = {}
    for f, (refv, vs) in levels.items():
        ref = F[f] == refv if f != 'maxRun' else F[f] <= refv
        res['levels'][f] = {'ref': refv, 'share': {int(v): float((F[f] == v).mean()) for v in np.unique(F[f])},
                            **{str(v): mh((F[f] == v) if v < max(vs) else (F[f] >= v), ref) for v in vs if ((F[f] == v) if v < max(vs) else (F[f] >= v)).sum() >= 200}}
    out[f'first{K}'] = res

# Exploratory (not in the preset plan): (1) death within 10 moves against the first 10 cards, closer to step 1's
# "death in the next 10 moves"; (2) how much of the within-position variance the card features explain
# (position fixed effects + one-hot features, OLS).
F = feats(script[:, :10]); keep = dead
dead = ((keep > 0) & (np.array([r['moves'] for r in rows]) <= 10)).astype(float)
out['exploratory'] = {'death10': float(dead.mean()),
                      'within10': {'bonus2_vs_0to1': mh(F['nBonus'] >= 2, F['nBonus'] <= 1),
                                   'absD12_ge4_vs_le2': mh(F['absD12'] >= 4, F['absD12'] <= 2),
                                   'absD12_5_vs_le2': mh(F['absD12'] == 5, F['absD12'] <= 2),
                                   'maxRun_ge5_vs_le3': mh(F['maxRun'] >= 5, F['maxRun'] <= 3),
                                   'maxRun_6_vs_le3': mh(F['maxRun'] == 6, F['maxRun'] <= 3)}}
def within_r2(y, K):
    F = feats(script[:, :K]); X = np.column_stack([np.minimum(F[f], cap)[:, None] == np.arange(1, cap + 1)
                                                   for f, cap in (('nBonus', 2), ('absD12', 6), ('maxRun', 7), ('n3', 8))]).astype(float)
    n = np.bincount(pos, minlength=P)[:, None]
    dm = lambda a: a - (np.stack([np.bincount(pos, a[:, j], P) for j in range(a.shape[1])], 1) / n)[pos]
    Xd, yd = dm(X), dm(y[:, None].astype(float))[:, 0]
    beta = np.linalg.lstsq(Xd, yd, rcond=None)[0]
    return float(1 - ((yd - Xd @ beta) ** 2).sum() / (yd ** 2).sum())
out['exploratory']['withinR2'] = {'death30_first10': within_r2(keep, 10), 'death30_first20': within_r2(keep, 20),
                                  'death10_first10': within_r2(dead, 10)}
dead = keep
(L / 'results2.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
print(json.dumps({k: out[k] for k in ('replays', 'positions', 'death30', 'variance', 'exploratory')}, ensure_ascii=False, indent=1))
