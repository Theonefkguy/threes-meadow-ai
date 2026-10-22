"""V49 step 3: a point system instead of an ordered list. Each legal direction gets
sum_f w_f * feature_f and the highest total is played. Weights: softmax regression over the legal
directions toward the AI's near-best set (gap < 0.25%), fitted on the even-seed V46 games; features
are added greedily (forward selection by test-free train "near"), then the weights are rounded to
small integers on a common scale to give a hand-usable score card.
  uv run --no-project --python 3.12 --with numpy python score.py"""
import json
import numpy as np
from pathlib import Path
L = Path(__file__).resolve().parent
z = np.load(L / 'cache.npz'); X, V, src, game, maxRank, FEAT = z['X'], z['V'], z['src'], z['game'], z['maxRank'], list(z['features'])
legal = ~np.isnan(V); keep = legal.sum(1) >= 2
X, V, src, game, maxRank, legal = X[keep], V[keep], src[keep], game[keep], maxRank[keep], legal[keep]
with np.errstate(invalid='ignore'):
    top = np.nanmax(V, 1); gap = (top[:, None] - V) / np.abs(top)[:, None]
NEAR, BAD = np.nan_to_num(gap) < 0.0025, np.nan_to_num(gap) >= 0.05
NEAR &= legal; BAD &= legal
best = np.nanargmax(V, 1)
X = np.nan_to_num(X.astype(np.float64))
train = (src == 'ai') & (game % 2 == 0); test = (src == 'ai') & (game % 2 == 1)
PHASES = [('≤192', 0, 9), ('384–768', 10, 11), ('1536', 12, 12), ('3072', 13, 13), ('6144+', 14, 99)]
scale = X[train][legal[train]].std(0) + 1e-9

def fit(fs, rows, iters=300, lr=0.5):
    x = X[rows][:, :, fs] / scale[fs]; m = legal[rows]; t = NEAR[rows] / NEAR[rows].sum(1, keepdims=True)
    w = np.zeros(len(fs)); g2 = np.zeros(len(fs))
    for _ in range(iters):
        s = np.where(m, x @ w, -np.inf); s -= s.max(1, keepdims=True); p = np.exp(s); p /= p.sum(1, keepdims=True)
        grad = ((p - t)[:, :, None] * x).sum((0, 1)) / len(x)
        g2 += grad ** 2; w -= lr * grad / np.sqrt(g2 + 1e-12)
    return w / scale[fs]
def evaluate(fs, w, rows):
    s = np.where(legal[rows], X[rows][:, :, fs] @ w, -np.inf); k = np.isclose(s, s.max(1, keepdims=True), rtol=0, atol=1e-9) & legal[rows]
    wk = k / k.sum(1, keepdims=True); n = len(wk)
    return {'n': int(n), 'near': float((wk * NEAR[rows]).sum() / n), 'match': float(wk[np.arange(n), best[rows]].sum() / n), 'bad5': float((wk * BAD[rows]).sum() / n)}

rng = np.random.default_rng(49); sub = np.flatnonzero(train); sub = rng.choice(sub, 60000, replace=False)
chosen, path = [], []
for step in range(10):
    cand = []
    for f in range(len(FEAT)):
        if f in chosen: continue
        fs = chosen + [f]; w = fit(fs, sub, iters=150); cand.append((evaluate(fs, w, sub)['near'], f))
    near, f = max(cand); chosen.append(f)
    w = fit(chosen, np.flatnonzero(train))
    path.append({'features': [FEAT[i] for i in chosen], 'weights': w.tolist(), 'train': evaluate(chosen, w, train), 'test': evaluate(chosen, w, test),
                 'v47': evaluate(chosen, w, src == 'v47'), 'v48': evaluate(chosen, w, src == 'v48')})
    p = path[-1]
    print(f"{step + 1:2d} +{FEAT[f]:10s} test near {p['test']['near']:.1%} bad5 {p['test']['bad5']:.1%} | v47 {p['v47']['near']:.1%} {p['v47']['bad5']:.1%} | v48 {p['v48']['near']:.1%} {p['v48']['bad5']:.1%}")
allw = fit(list(range(len(FEAT))), np.flatnonzero(train))
print('all', len(FEAT), evaluate(list(range(len(FEAT))), allw, test), evaluate(list(range(len(FEAT))), allw, src == 'v48'))
(L / 'score.json').write_text(json.dumps({'path': path, 'all': {'weights': dict(zip(FEAT, allw.tolist())), 'test': evaluate(list(range(len(FEAT))), allw, test)}}, indent=1) + '\n')
