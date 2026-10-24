"""V57 step 5: does a view work out the board after its own move? (as V52 simulate.py, per view)
For the 4 views of 2000 validation positions, cell token i after layer l -> what happens to cell i of the
turned board under each of the 4 directions: stays / gets the card behind it / becomes the merged card /
becomes empty. A 4-way ridge classifier per direction and layer, held-out games as V51.
Left is the move this view judges; right / up / down are moves the view does not judge.
Baseline: view one-hot + preview + this cell's index and rank.  -> simulate.json"""
import json
import numpy as np
from common import load_net, sample, to_dev, run, rotate, L
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
    return [0 if after[i] == b[i] else 3 if after[i] == 0 else 1 if up[i] is not None and after[i] == b[up[i]] else 2 for i in range(16)]

if __name__ == '__main__':
    net = load_net(); a = sample(); m = 2000
    _, _, _, res, _, _ = run(net, to_dev(a, slice(0, m)))
    C = np.stack([r[:, 1:17].cpu().numpy().reshape(4, m, 16, -1).transpose(1, 0, 2, 3).reshape(4 * m, 16, -1) for r in res], 1)   # [4m,5,16,d]
    vb = rotate(net, a['board'][:m].astype(int)).reshape(-1, 16)                                                     # [4m,16], position-major
    test_games = np.random.default_rng(5).choice(np.unique(a['seed']), 30, replace=False)
    te = np.repeat(np.repeat(np.isin(a['seed'][:m], test_games), 4), 16); tr = ~te
    onehot = np.c_[np.eye(16)[vb].reshape(len(vb), -1), np.repeat(np.eye(17)[a['cards'][:m]].sum(1), 4, 0)]
    base = np.c_[np.repeat(onehot, 16, 0), np.tile(np.eye(16), (len(vb), 1)), np.eye(16)[vb.reshape(-1)]]
    out = {}
    for d, name in enumerate(['往左（本视角判断的一步）', '往右', '往上', '往下']):
        Y = np.array([kinds(b, d) for b in vb.tolist()]).reshape(-1); ok = np.repeat(np.array([bool(F49.move(b, d)[2]) for b in vb.tolist()]), 16)
        out[name] = {'baseline': fit_acc(base, Y, tr, te, ok), 'layers': [fit_acc(C[:, l].reshape(-1, C.shape[-1]), Y, tr, te, ok) for l in range(5)],
                     'share': {k: float((Y[te & ok] == c).mean()) for c, k in enumerate(KINDS)}}
        print(name, 'baseline', {k: round(v, 3) for k, v in out[name]['baseline'].items()}, flush=True)
        for l in range(5): print('   layer', l, {k: round(v, 3) for k, v in out[name]['layers'][l].items()}, flush=True)
    (L / 'simulate.json').write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n')
