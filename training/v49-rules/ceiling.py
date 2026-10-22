"""V49 step 4: how much of the AI's choice the V49 features can explain at all. A small network
(features of one direction + phase -> 64 -> 64 -> score, shared by the four directions, softmax over
the legal ones toward the AI's near-best set) trained on the even-seed V46 games. If it is far above
the rule lists, the rule format is the limit; if not, the features are.
  uv run --no-project --python 3.12 --with numpy python ceiling.py"""
import json
import numpy as np
from pathlib import Path
L = Path(__file__).resolve().parent
z = np.load(L / 'cache.npz'); X, V, src, game, maxRank, FEAT = z['X'], z['V'], z['src'], z['game'], z['maxRank'], list(z['features'])
legal = ~np.isnan(V); keep = legal.sum(1) >= 2
X, V, src, game, maxRank, legal = X[keep], V[keep], src[keep], game[keep], maxRank[keep], legal[keep]
with np.errstate(invalid='ignore'):
    top = np.nanmax(V, 1); gap = np.nan_to_num((top[:, None] - V) / np.abs(top)[:, None], nan=9)
NEAR, BAD = (gap < 0.0025) & legal, (gap >= 0.05) & legal
print('positions without a defined gap (best value 0), left out of training:', int((~NEAR.any(1)).sum()))
X = np.nan_to_num(X.astype(np.float32))
X = np.concatenate([X, np.repeat((maxRank[:, None, None] / 14).astype(np.float32), 4, 1)], 2)
train = (src == 'ai') & (game % 2 == 0); test = (src == 'ai') & (game % 2 == 1)
mu, sd = X[train][legal[train]].mean(0), X[train][legal[train]].std(0) + 1e-6
X = (X - mu) / sd
rng = np.random.default_rng(49); H = 64
P = {'w1': rng.normal(0, 1 / np.sqrt(X.shape[2]), (X.shape[2], H)), 'b1': np.zeros(H), 'w2': rng.normal(0, 1 / np.sqrt(H), (H, H)), 'b2': np.zeros(H),
     'w3': rng.normal(0, 1 / np.sqrt(H), (H, 1)), 'b3': np.zeros(1)}
M = {k: np.zeros_like(v) for k, v in P.items()}; S = {k: np.zeros_like(v) for k, v in P.items()}

def forward(x):
    h1 = np.maximum(0, x @ P['w1'] + P['b1']); h2 = np.maximum(0, h1 @ P['w2'] + P['b2']); return (h2 @ P['w3'] + P['b3'])[..., 0], (h1, h2)
def evaluate(rows):
    s, _ = forward(X[rows]); s = np.where(legal[rows], s, -np.inf); k = (s == s.max(1, keepdims=True)) & legal[rows]; n = len(s)
    w = k / k.sum(1, keepdims=True)
    return {'n': int(n), 'near': float((w * NEAR[rows]).sum() / n), 'bad5': float((w * BAD[rows]).sum() / n)}
idx = np.flatnonzero(train & NEAR.any(1)); t_step = 0
for epoch in range(6):
    rng.shuffle(idx)
    for i in range(0, len(idx), 512):
        b = idx[i:i + 512]; x = X[b]; m = legal[b]; tgt = NEAR[b] / NEAR[b].sum(1, keepdims=True)
        s, (h1, h2) = forward(x); s = np.where(m, s, -1e9); s -= s.max(1, keepdims=True); p = np.exp(s) * m; p /= p.sum(1, keepdims=True)
        g = (p - tgt) / len(b)                                   # d loss / d score
        G = {'w3': np.einsum('bdh,bd->h', h2, g)[:, None], 'b3': np.array([g.sum()])}
        d2 = (g[..., None] * P['w3'][:, 0]) * (h2 > 0); G['w2'] = np.einsum('bdi,bdh->ih', h1, d2); G['b2'] = d2.sum((0, 1))
        d1 = (d2 @ P['w2'].T) * (h1 > 0); G['w1'] = np.einsum('bdi,bdh->ih', x, d1); G['b1'] = d1.sum((0, 1))
        t_step += 1
        for k in P:
            M[k] = 0.9 * M[k] + 0.1 * G[k]; S[k] = 0.999 * S[k] + 0.001 * G[k] ** 2
            P[k] -= 1e-3 * (M[k] / (1 - 0.9 ** t_step)) / (np.sqrt(S[k] / (1 - 0.999 ** t_step)) + 1e-8)
    print(epoch, 'train', evaluate(np.flatnonzero(train)[:50000]), 'test', evaluate(np.flatnonzero(test)))
out = {s: evaluate(np.flatnonzero(r)) for s, r in (('test', test), ('v47', src == 'v47'), ('v48', src == 'v48'))}
print(out)
(L / 'ceiling.json').write_text(json.dumps(out, indent=1) + '\n')
