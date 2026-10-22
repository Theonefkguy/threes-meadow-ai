"""V49 step 5: each rule on its own, and the 4-rule list on the AI's clear-cut positions (second-best
direction >= 2% worse), against what the humans of V47 / V48 did on their clear-cut positions.
  uv run --no-project --python 3.12 --with numpy python summary.py"""
import json
import numpy as np
from pathlib import Path
L = Path(__file__).resolve().parent
z = np.load(L / 'cache.npz'); X, V, src, game, FEAT = z['X'], z['V'], z['src'], z['game'], list(z['features'])
legal = ~np.isnan(V); keep = legal.sum(1) >= 2
X, V, src, game, legal = X[keep], V[keep], src[keep], game[keep], legal[keep]
with np.errstate(invalid='ignore'):
    top = np.nanmax(V, 1); gap = np.nan_to_num((top[:, None] - V) / np.abs(top)[:, None], nan=9)
NEAR, BAD = (gap < 0.0025) & legal, (gap >= 0.05) & legal
best = np.argmax(np.where(legal, V, -np.inf), 1); clear = np.sort(np.where(legal, gap, 9), 1)[:, 1] >= 0.02
R = [(FEAT.index(r['feature']), r['sign']) for r in json.loads((L / 'rules.json').read_text())['rules']['all']]
def play(rules, rows):
    k = legal[rows].copy()
    for f, s in rules:
        v = np.where(k, s * X[rows][:, :, f], -np.inf); k &= v == v.max(1, keepdims=True)
    return k / k.sum(1, keepdims=True)
def m(rules, rows):
    w = play(rules, rows); n = rows.sum(); c = clear[rows]
    return {'near': float((w * NEAR[rows]).sum() / n), 'bad5': float((w * BAD[rows]).sum() / n),
            'clearMatch': float(w[np.arange(n), best[rows]][c].mean())}
test = (src == 'ai') & (game % 2 == 1)
out = {'alone': {FEAT[f]: m([(f, s)], test) for f, s in R}, 'list': {n: {k: m(R[:n], r) for k, r in (('test', test), ('v47', src == 'v47'), ('v48', src == 'v48'))} for n in range(len(R) + 1)}}
(L / 'summary.json').write_text(json.dumps(out, indent=1) + '\n')
for k, v in out['alone'].items(): print('alone', k, {a: round(b, 3) for a, b in v.items()})
for n, v in out['list'].items(): print(n, {k: {a: round(b, 3) for a, b in x.items()} for k, x in v.items()})
