"""V51 step 3: linear probes. Ridge regression (penalty chosen on 20% of the training games) from the
residual stream after each layer to quantities of the position; R^2 on held-out games (30 of the 100
validation games). Baseline: the same regression on the raw input as one-hot (board 16x16, preview).
Only what a layer predicts clearly better than the baseline is something the model has computed.
- summary token (layers 1-4; it is a constant before layer 1): empties, adjacent mergeable pairs,
  largest card in a corner, chain from the largest card (V49), legal moves, log(1 + moves until the AI's
  game shows a larger top card), log of the AI's best value, the AI's best direction (accuracy);
  each V49 feature of each legal direction (R^2 averaged over the 4 directions);
- cell tokens (layers 0-4, one probe shared by all 16 cells, 8000 positions): this cell has an adjacent
  merge partner; this cell is stuck (no adjacent empty cell or partner); this cell is the largest card;
  this cell's card moves under a direction (averaged over the 4 directions).
  Cell baseline: board one-hot + preview + this cell's index and rank one-hot.
  -> probe.json, fig/probe.png"""
import json, sys
from multiprocessing import Pool
import numpy as np
from common import load_net, sample, to_dev, run, plt_setup, L
sys.path.insert(0, str(L.parent / 'v49-rules'))
import features as F49

def feats(args):
    b, c = args; return F49.features(b, [x for x in c if x])
def moved_sets(b): return [F49.move(b, d)[1] for d in range(4)]

def ridge_r2(X, y, tr, te, binary=False):
    ok = ~np.isnan(y); tr, te = tr & ok, te & ok
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Z = (X - mu) / sd; Z = np.c_[Z, np.ones(len(Z))]
    idx = np.flatnonzero(tr); rng = np.random.default_rng(0); inner = rng.random(len(idx)) < 0.8
    best = None
    for lam in (0.1, 10, 1000):
        A = Z[idx[inner]]; w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[idx[inner]])
        e = ((Z[idx[~inner]] @ w - y[idx[~inner]]) ** 2).mean()
        if best is None or e < best[0]: best = (e, lam)
    A = Z[tr]; w = np.linalg.solve(A.T @ A + best[1] * np.eye(A.shape[1]), A.T @ y[tr]); p = Z[te] @ w
    r2 = 1 - ((p - y[te]) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
    return float(r2) if not binary else (float(r2), float(((p > 0.5) == (y[te] > 0.5)).mean()), float(max(y[te].mean(), 1 - y[te].mean())))

def classify_acc(X, lab, tr, te):
    Y = np.eye(4)[lab]; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    w = np.linalg.solve(Z[tr].T @ Z[tr] + 10 * np.eye(Z.shape[1]), Z[tr].T @ Y[tr]); return float(((Z[te] @ w).argmax(1) == lab[te]).mean())

if __name__ == '__main__':
    net = load_net(); a = sample(); n = len(a['board']); B = a['board'].astype(int)
    games = np.unique(a['seed']); test_games = np.random.default_rng(5).choice(games, 30, replace=False)
    te = np.isin(a['seed'], test_games); tr = ~te
    # activations: summary token and cell tokens after each layer
    S, Cells = [], []
    for i in range(0, n, 4096):
        _, res, _, _ = run(net, to_dev(a, slice(i, i + 4096)))
        S.append(np.stack([r[:, 0].cpu().numpy() for r in res], 1)); Cells.append(np.stack([r[:, 1:17].cpu().numpy() for r in res], 1))
    S = np.concatenate(S); Cells = np.concatenate(Cells)                        # [n,5,128], [n,5,16,128]
    onehot = np.c_[np.eye(16)[B].reshape(n, -1), np.eye(17)[a['cards']].sum(1)]
    # targets
    with Pool(12) as p: V49 = np.stack(p.map(feats, list(zip(B.tolist(), a['cards'].tolist())), chunksize=500))   # [n,4,22]
    mergeable = lambda x, y: (x > 0) & (y > 0) & (((x + y) == 3) | ((x >= 3) & (x == y)))
    pairs = sum(mergeable(B[:, i], B[:, j]) for i, j in F49.ADJ)
    v = a['values'].astype(float); best = np.nanmax(v, 1)
    pos = {'空格数': (B == 0).sum(1), '可合并的相邻对数': pairs, '最大牌在角上': np.isin(B.argmax(1), [0, 3, 12, 15]).astype(float),
           '从最大牌出发的链长': np.array([F49.chain(b) for b in B.tolist()]), '合法方向数': a['legal'].sum(1),
           '再过几步出现更大的牌 (log)': np.log1p(a['nextMax']), 'AI 估分 (log)': np.log(np.maximum(best, 1))}
    lab = np.nan_to_num(v, nan=-np.inf).argmax(1)
    out = {'positions': n, 'testGames': 30, 'summary': {}, 'directions': {}, 'cells': {}}
    for k, y in pos.items():
        y = y.astype(float); bin_ = set(np.unique(y[~np.isnan(y)])) <= {0.0, 1.0}
        out['summary'][k] = {'baseline': ridge_r2(onehot, y, tr, te, bin_), 'layers': [ridge_r2(S[:, l], y, tr, te, bin_) for l in range(1, 5)]}
        print(k, out['summary'][k], flush=True)
    out['summary']['AI 首选方向（正确率）'] = {'baseline': classify_acc(onehot, lab, tr, te), 'layers': [classify_acc(S[:, l], lab, tr, te) for l in range(1, 5)]}
    print('best dir', out['summary']['AI 首选方向（正确率）'], flush=True)
    for f, name in enumerate(F49.FEATURES):
        r = {'baseline': np.mean([ridge_r2(onehot, V49[:, d, f], tr, te) for d in range(4)]),
             'layers': [float(np.mean([ridge_r2(S[:, l], V49[:, d, f], tr, te) for d in range(4)])) for l in range(1, 5)]}
        out['directions'][name] = r; print(name, {k: np.round(v, 3) for k, v in r.items()}, flush=True)
    # cell probes: 8000 positions x 16 cells
    m = 8000; Bm = B[:m]; trc = np.repeat(tr[:m], 16); tec = np.repeat(te[:m], 16)
    nb = F49.NB
    canm = np.array([[any(mergeable(b[i], b[j]) for j in nb[i]) for i in range(16)] for b in Bm]).astype(float)
    stuck = np.array([[b[i] > 0 and not any(b[j] == 0 or mergeable(b[i], b[j]) for j in nb[i]) for i in range(16)] for b in Bm]).astype(float)
    ismax = np.zeros((m, 16)); ismax[np.arange(m), Bm.argmax(1)] = 1
    mv = [moved_sets(b) for b in Bm.tolist()]
    moves = [np.array([[float(i in ms[d]) if Bm[k, i] else np.nan for i in range(16)] for k, ms in enumerate(mv)]) for d in range(4)]
    cbase = np.c_[np.repeat(onehot[:m], 16, 0), np.tile(np.eye(16), (m, 1)), np.eye(16)[Bm.reshape(-1)]]
    cells = {'这一格有相邻的可合并牌': canm, '这一格被困住': stuck, '这一格是最大牌': ismax}
    for k, y in cells.items():
        y = y.reshape(-1)
        out['cells'][k] = {'baseline': ridge_r2(cbase, y, trc, tec, True), 'layers': [ridge_r2(Cells[:m, l].reshape(-1, 128), y, trc, tec, True) for l in range(5)]}
        print(k, out['cells'][k], flush=True)
    r = {'baseline': [ridge_r2(cbase, moves[d].reshape(-1), trc, tec, True) for d in range(4)],
         'layers': [[ridge_r2(Cells[:m, l].reshape(-1, 128), moves[d].reshape(-1), trc, tec, True) for d in range(4)] for l in range(5)]}
    out['cells']['这一格的牌在某方向上会移动'] = {'baseline': tuple(np.mean(r['baseline'], 0)), 'layers': [tuple(np.mean(x, 0)) for x in r['layers']]}
    print('moves', out['cells']['这一格的牌在某方向上会移动'], flush=True)
    (L / 'probe.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float) + '\n')
