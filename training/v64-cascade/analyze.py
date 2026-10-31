"""V64 analysis (setup in RL-V64-CASCADE.md): stop the website search after depth 4 or 5 when a judge says that
iteration already picks the depth-6 move. Judges: A = root margin thresholds, B = small networks. Thresholds are
chosen on training games (B: networks fit on 500 games, thresholds on the other 100), reported on the 200 test
games: share of moves whose move changes, share of the 6-ply search time saved, and the 6-ply value lost by the
changed moves.  -> results.json"""
import json
from pathlib import Path
import numpy as np
import torch
D = Path('/Volumes/Extreme SSD/Datasets/threes-v64'); L = Path(__file__).resolve().parent
torch.manual_seed(64); torch.set_num_threads(6)

A = np.concatenate([np.loadtxt(f) for f in sorted(D.glob('runs/c-*.txt'))]); A = A[np.argsort(A[:, 0])]
N = len(A); assert (A[:, 0] == np.arange(N)).all()
meta = np.load(D / 'meta.npz')
dirs = A[:, 1::6].astype(int); ms = A[:, 2::6]; V = np.stack([A[:, 3 + 6 * d:7 + 6 * d] for d in range(6)], 1)   # N x 6 x 4
legal = V[:, 5] > -1e299
dir6, t6 = dirs[:, 5], ms[:, 5]
out = {'positions': N, 'websiteMove': float((dir6 == meta['chosen']).mean()),
       'agree': {d + 1: float((dirs[:, d] == dir6).mean()) for d in range(5)},
       'cumulativeMsShare': {d + 1: float(ms[:, d].sum() / t6.sum()) for d in range(6)}, 'ms6': float(t6.mean())}

seeds = np.unique(meta['seed']); test_s = seeds[::4]; rest = np.setdiff1d(seeds, test_s); val_s = rest[::6]
test = np.isin(meta['seed'], test_s); val = np.isin(meta['seed'], val_s); fit = ~test & ~val; train = ~test

def margin(d):
    v = np.where(legal, V[:, d], -np.inf); s = np.sort(v, 1); best, second = s[:, -1], s[:, -2]
    return np.where(np.isfinite(second), (best - second) / np.maximum(np.abs(best), 1), np.inf)   # best can be 0
rel = lambda d: np.clip(np.where(legal, (V[:, d] - V[:, d].max(1, keepdims=True)) / np.maximum(np.abs(V[:, d].max(1, keepdims=True)), 1), -1), -1, 0)

def evaluate(stop4, stop5, sel):
    final = np.where(stop4, dirs[:, 3], np.where(stop5, dirs[:, 4], dir6)); tstop = np.where(stop4, ms[:, 3], np.where(stop5, ms[:, 4], t6))
    changed = final != dir6; v6 = V[:, 5]
    lost = (v6[np.arange(N), dir6] - v6[np.arange(N), final]) / np.maximum(np.abs(v6[np.arange(N), dir6]), 1)
    return {'changed': float(changed[sel].mean()), 'saved': float((t6 - tstop)[sel].sum() / t6[sel].sum()),
            'stop4': float(stop4[sel].mean()), 'stop5': float((~stop4 & stop5)[sel].mean()),
            'lostWhenChanged': float(lost[sel & changed].mean()) if (sel & changed).any() else 0.0}

def sweep(s4, s5, sel, grid4, grid5):
    """scores s4, s5 (higher = surer); all threshold pairs -> list of (thr4, thr5, metrics on sel)"""
    return [(a, b, evaluate(s4 >= a, s5 >= b, sel)) for a in grid4 for b in grid5]

def pick(rows, target):                       # max saving with changed <= target
    ok = [r for r in rows if r[2]['changed'] <= target]
    return max(ok, key=lambda r: r[2]['saved']) if ok else None

targets = [0.0025, 0.005, 0.01, 0.02, 0.05]
oracle4 = dirs[:, 3] == dir6; oracle5 = dirs[:, 4] == dir6
out['oracle'] = evaluate(oracle4, oracle5, test)

# A: root margin
m4, m5 = margin(3), margin(4)
g = lambda m: np.r_[np.quantile(m[train & np.isfinite(m)], np.linspace(0, 1, 41)[1:-1]), np.inf]
rowsA = sweep(m4, m5, train, g(m4), g(m5))
out['A'] = {}
for t in targets:
    r = pick(rowsA, t)
    if r: out['A'][t] = {'thr4': float(r[0]), 'thr5': float(r[1]), 'train': r[2], 'test': evaluate(m4 >= r[0], m5 >= r[1], test)}

# B: small networks P(depth-d move = depth-6 move)
pos = [l.split() for l in (D / 'positions.txt').read_text().splitlines()]
board = np.array([[int(x) for x in p[1:17]] for p in pos], float)
pre = np.array([[int(p[17]), int(p[18]), int(p[17 + int(p[17]) + 1])] for p in pos], float)   # ncards, first card, bonus flag
def feats(d):
    cols = [rel(l) for l in range(d + 1)] + [np.minimum(margin(l), 1)[:, None] for l in range(d + 1)]
    cols += [(dirs[:, l] == dirs[:, d])[:, None] for l in range(d)]
    cols += [legal.sum(1, keepdims=True) / 4, (board == 0).sum(1, keepdims=True) / 16, board.max(1, keepdims=True) / 15,
             board / 15, pre / np.array([3, 15, 1])]
    X = np.concatenate(cols, 1).astype(np.float32); assert np.isfinite(X).all(); return X
def net(X, y):
    Xt, yt = torch.tensor(X[fit]), torch.tensor(y[fit], dtype=torch.float32)
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    f = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 128), torch.nn.ReLU(), torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1))
    opt = torch.optim.Adam(f.parameters(), 1e-3)
    for epoch in range(30):
        for i in torch.randperm(len(Xt)).split(512):
            opt.zero_grad(); torch.nn.functional.binary_cross_entropy_with_logits(f((Xt[i] - mu) / sd)[:, 0], yt[i]).backward(); opt.step()
    nets.append((mu, sd, f))
    with torch.no_grad(): return torch.sigmoid(f((torch.tensor(X) - mu) / sd)[:, 0]).numpy()
nets = []; p4, p5 = net(feats(3), oracle4), net(feats(4), oracle5)
def aucf(s, y):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
out['auc'] = {'margin4': aucf(m4[test], oracle4[test]), 'margin5': aucf(m5[test], oracle5[test]),
              'net4': aucf(p4[test], oracle4[test]), 'net5': aucf(p5[test], oracle5[test])}
gp = np.r_[1 - np.logspace(-4, np.log10(0.5), 40), 2.0]
rowsB = sweep(p4, p5, val, gp, gp)
out['B'] = {}
for t in targets:
    r = pick(rowsB, t)
    if r: out['B'][t] = {'thr4': float(r[0]), 'thr5': float(r[1]), 'val': r[2], 'test': evaluate(p4 >= r[0], p5 >= r[1], test)}
# the 1% operating point for the whole-game test (V65): judge.txt = thr4 thr5, then per network: inputs, mu, sd,
# W1 b1 W2 b2 W3 b3 (torch layout, out x in); network probabilities kept for checking the C++ copy
with open(L / 'judge.txt', 'w') as fo:
    fo.write('%.9g %.9g\n' % (out['B'][0.01]['thr4'], out['B'][0.01]['thr5']))
    for mu, sd, f in nets:
        fo.write('%d\n' % len(mu))
        for t in (mu, sd, *[q for l in (0, 2, 4) for q in (f[l].weight, f[l].bias)]): fo.write(' '.join('%.9g' % x for x in t.detach().reshape(-1).tolist()) + '\n')
np.savez(D / 'judge-probs.npz', p4=p4, p5=p5)
(L / 'results.json').write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps({k: out[k] for k in ('positions', 'websiteMove', 'agree', 'cumulativeMsShare', 'ms6', 'oracle', 'auc')}, indent=1))
for j in ('A', 'B'):
    for t, r in out[j].items(): print(j, t, 'test changed %.4f saved %.3f stop4 %.3f stop5 %.3f lost %.4f' % tuple(r['test'][k] for k in ('changed', 'saved', 'stop4', 'stop5', 'lostWhenChanged')))
