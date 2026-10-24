"""V57 step 3: linear probes per view (ridge regression as V51 probe.py, penalty chosen on 20% of the
training games; R^2 or accuracy on 30 held-out games). Baseline: the same regression on the view's raw
input (turned board one-hot 16x16 + preview multi-hot).
- summary token of a legal view (layers 1-4): the 22 V49 features of this move (left on the turned
  board = the view's direction on the real board); the AI's value of this move (log); its gap to the AI's
  best (min(gap, 0.1)); whether it is the AI's best; whole-position quantities (empties, adjacent mergeable
  pairs, chain from the largest card, log(1 + moves until a larger top card), log of the AI's best value);
- cell tokens of the views of 2000 positions (layers 0-4, one probe for all cells): this cell has an
  adjacent merge partner; its left neighbour is a merge partner; it moves when this view pushes left;
  it moves when pushing right / up / down (on the turned board); it is stuck; it is the largest card.
  Cell baseline: view one-hot + preview + this cell's index and rank.  -> probe.json"""
import json
from multiprocessing import Pool
import numpy as np
import torch
from common import load_net, sample, to_dev, run, rotate, gaps, L
import features as F49

def ridge_r2(X, y, tr, te, binary=False):
    ok = ~np.isnan(y); tr, te = tr & ok, te & ok
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Z = (X - mu) / sd; Z = np.c_[Z, np.ones(len(Z))]
    idx = np.flatnonzero(tr); inner = np.random.default_rng(0).random(len(idx)) < 0.8; best = None
    for lam in (0.1, 10, 1000):
        A = Z[idx[inner]]; w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[idx[inner]])
        e = ((Z[idx[~inner]] @ w - y[idx[~inner]]) ** 2).mean()
        if best is None or e < best[0]: best = (e, lam)
    A = Z[tr]; w = np.linalg.solve(A.T @ A + best[1] * np.eye(A.shape[1]), A.T @ y[tr]); p = Z[te] @ w
    r2 = 1 - ((p - y[te]) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
    return float(r2) if not binary else (float(r2), float(((p > 0.5) == (y[te] > 0.5)).mean()), float(max(y[te].mean(), 1 - y[te].mean())))
def feats(x): return F49.features(x[0], [c for c in x[1] if c])
mergeable = lambda x, y: (x > 0) & (y > 0) & (((x + y) == 3) | ((x >= 3) & (x == y)))

if __name__ == '__main__':
    net = load_net(); a = sample(); n = len(a['board']); B = a['board'].astype(int); VB = rotate(net, B)          # [n,4,16]
    test_games = np.random.default_rng(5).choice(np.unique(a['seed']), 30, replace=False); te_pos = np.isin(a['seed'], test_games)
    S, Cells = [], []                                                                                          # [n,4,5,128], [m,4,5,16,128]
    for i in range(0, n, 2048):
        _, _, _, res, _, _ = run(net, to_dev(a, slice(i, i + 2048))); k = min(i + 2048, n) - i
        S.append(np.stack([r[:, 0].cpu().numpy().reshape(4, k, -1).transpose(1, 0, 2) for r in res], 2))
        if i < 2000: Cells.append(np.stack([r[:, 1:17].cpu().numpy().reshape(4, k, 16, -1).transpose(1, 0, 2, 3) for r in res], 2))
    S = np.concatenate(S); Cells = np.concatenate(Cells)[:2000]
    legal = a['legal']; lv = legal.reshape(-1)                                                                 # views as (position, direction)
    flat = lambda z: z.reshape(n * 4, *z.shape[2:])[lv]
    te = np.repeat(te_pos, 4)[lv]; tr = ~te
    onehot = np.c_[np.eye(16)[VB].reshape(n, 4, -1), np.repeat(np.eye(17)[a['cards']].sum(1)[:, None], 4, 1)]
    X0 = flat(onehot); SL = flat(S)                                                                           # [v,273], [v,5,128]
    with Pool(12) as p: V49 = np.stack(p.map(feats, list(zip(B.tolist(), a['cards'].tolist())), chunksize=500))    # [n,4,22]
    v = a['values'].astype(float); g = gaps(a['values']); best = np.nanmax(v, 1)
    with np.errstate(divide='ignore', invalid='ignore'): logv = np.where(legal, np.log(np.maximum(v, 1)), np.nan)
    per_view = {'这一步：AI 估分 (log)': logv, '这一步：比首选差多少 (≤10%)': np.where(legal, np.minimum(g, 0.1), np.nan),
                '这一步：是 AI 首选': (g == 0).astype(float)}
    pairs = sum(mergeable(B[:, i], B[:, j]) for i, j in F49.ADJ)
    whole = {'空格数': (B == 0).sum(1), '可合并的相邻对数': pairs, '从最大牌出发的链长': np.array([F49.chain(b) for b in B.tolist()]),
             '再过几步出现更大的牌 (log)': np.log1p(a['nextMax']), 'AI 估分 (log)': np.log(np.maximum(best, 1))}
    out = {'positions': n, 'views': int(lv.sum()), 'summary': {}, 'thisMove': {}, 'cells': {}}
    def probe(y, binary=False):
        return {'baseline': ridge_r2(X0, y, tr, te, binary), 'layers': [ridge_r2(SL[:, l], y, tr, te, binary) for l in range(1, 5)]}
    for k, y in {**per_view, **{k: np.repeat(y[:, None].astype(float), 4, 1) for k, y in whole.items()}}.items():
        y = flat(y); bin_ = set(np.unique(y[~np.isnan(y)])) <= {0.0, 1.0}
        out['summary'][k] = probe(y, bin_); print(k, out['summary'][k], flush=True)
    for f, name in enumerate(F49.FEATURES):
        out['thisMove'][name] = probe(flat(V49[:, :, f].astype(float))); print(name, out['thisMove'][name], flush=True)
    # cell probes: the 4 views of 2000 positions x 16 cells
    m = 2000; vb = VB[:m].reshape(-1, 16); nv = len(vb)
    tec = np.repeat(np.repeat(te_pos[:m], 4), 16); trc = ~tec
    nb = F49.NB
    canm = np.array([[any(mergeable(b[i], b[j]) for j in nb[i]) for i in range(16)] for b in vb]).astype(float)
    left = np.array([[bool(i % 4) and bool(mergeable(b[i], b[i - 1])) for i in range(16)] for b in vb]).astype(float)
    stuck = np.array([[b[i] > 0 and not any(b[j] == 0 or mergeable(b[i], b[j]) for j in nb[i]) for i in range(16)] for b in vb]).astype(float)
    ismax = np.zeros((nv, 16)); ismax[np.arange(nv), vb.argmax(1)] = 1
    moves = []
    for d in range(4):
        ms = [F49.move(b, d)[1] for b in vb.tolist()]
        moves.append(np.array([[float(i in s) if vb[k, i] else np.nan for i in range(16)] for k, s in enumerate(ms)]))
    cbase = np.c_[np.repeat(onehot[:m].reshape(nv, -1), 16, 0), np.tile(np.eye(16), (nv, 1)), np.eye(16)[vb.reshape(-1)]]
    Cv = Cells.reshape(nv, 5, 16, -1)
    cells = {'这一格有相邻的可合并牌': canm, '左边紧挨着可合并的牌': left, '往左推会动（本视角要判断的一步）': moves[0],
             '往右推会动': moves[1], '往上推会动': moves[2], '往下推会动': moves[3], '这一格被困住': stuck, '这一格是最大牌': ismax}
    for k, y in cells.items():
        y = y.reshape(-1)
        out['cells'][k] = {'baseline': ridge_r2(cbase, y, trc, tec, True), 'layers': [ridge_r2(Cv[:, l].reshape(-1, Cv.shape[-1]), y, trc, tec, True) for l in range(5)]}
        print(k, out['cells'][k], flush=True)
    (L / 'probe.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float) + '\n')
