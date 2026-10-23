"""V52 question 1: does the model work out the board after each move?
For 8000 validation positions, cell token i after layer l -> what happens to cell i when moving in
direction d: stays the same / gets the card behind it (slide) / becomes the merged card / becomes empty.
A 4-way linear classifier (ridge on one-hot targets), one per direction and layer, held-out games as in
V51. Reported: accuracy, and recall of each kind (averaged over directions).
(An exact 16-way "which card ends up here" probe was tried first: no layer gets the changed cells right
more than 40% of the time, merged cells under 8%; the card that arrives is not linearly readable.)
Baseline: board one-hot + preview + this cell's index and rank.  -> simulate.json"""
import json, sys
from pathlib import Path
import numpy as np
L = Path(__file__).resolve().parent; sys.path.insert(0, str(L.parent / 'v51-interpret')); sys.path.insert(0, str(L.parent / 'v49-rules'))
from common import load_net, sample, to_dev, run
import features as F49

KINDS = ['不变', '被推进来', '合并点', '变空']
def fit_acc(X, Y, tr, te, ok):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    T = np.eye(4)[Y]; w = np.linalg.solve(Z[tr & ok].T @ Z[tr & ok] + 10 * np.eye(Z.shape[1]), Z[tr & ok].T @ T[tr & ok]); p = (Z @ w).argmax(1)
    t = te & ok
    return {'accuracy': float((p[t] == Y[t]).mean()), **{k: float((p[t & (Y == c)] == c).mean()) for c, k in enumerate(KINDS)}}

def kinds(b, d):
    after = F49.move(b, d)[0]; up = [None] * 16
    for idx in F49.LANES[d]:
        for j in range(3): up[idx[j]] = idx[j + 1]
    k = []
    for i in range(16):
        if after[i] == b[i]: k.append(0)
        elif after[i] == 0: k.append(3)
        elif up[i] is not None and after[i] == b[up[i]]: k.append(1)
        else: k.append(2)
    return k

if __name__ == '__main__':
    net = load_net(); a = sample(); m = 8000; B = a['board'][:m].astype(int)
    games = np.unique(a['seed']); test_games = np.random.default_rng(5).choice(games, 30, replace=False)
    te = np.repeat(np.isin(a['seed'][:m], test_games), 16); tr = ~te
    C = []
    for i in range(0, m, 4096):
        _, res, _, _ = run(net, to_dev(a, slice(i, min(i + 4096, m)))); C.append(np.stack([r[:, 1:17].cpu().numpy() for r in res], 1))
    C = np.concatenate(C)                                                  # [m,5,16,128]
    onehot = np.c_[np.eye(16)[B].reshape(m, -1), np.eye(17)[a['cards'][:m]].sum(1)]
    base = np.c_[np.repeat(onehot, 16, 0), np.tile(np.eye(16), (m, 1)), np.eye(16)[B.reshape(-1)]]
    out = {}
    for d in range(4):
        Y = np.array([kinds(b, d) for b in B.tolist()]).reshape(-1); ok = np.repeat(np.array([bool(F49.move(b, d)[2]) for b in B.tolist()]), 16)
        r = {'baseline': fit_acc(base, Y, tr, te, ok), 'layers': [fit_acc(C[:, l].reshape(-1, 128), Y, tr, te, ok) for l in range(5)],
             'share': {k: float((Y[te & ok] == c).mean()) for c, k in enumerate(KINDS)}}
        out[d] = r; print(d, r['layers'][3], flush=True)
    keys = ['accuracy'] + KINDS
    avg = {'share': {k: float(np.mean([out[d]['share'][k] for d in range(4)])) for k in KINDS},
           'baseline': {k: float(np.mean([out[d]['baseline'][k] for d in range(4)])) for k in keys},
           'layers': [{k: float(np.mean([out[d]['layers'][l][k] for d in range(4)])) for k in keys} for l in range(5)]}
    print('average', json.dumps(avg, ensure_ascii=False))
    (L / 'simulate.json').write_text(json.dumps({'byDirection': out, 'average': avg}, indent=1, ensure_ascii=False) + '\n')
