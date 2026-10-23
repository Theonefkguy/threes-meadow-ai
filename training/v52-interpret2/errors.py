"""V52 question 2: where the big mistakes are and whether the model knows it is unsure.
30000 validation positions. bad5 = chosen direction >= 5% worse than the AI's best (root values).
- bad5 rate by phase (top card);
- in bad5 positions: V49 features of the chosen direction minus those of the AI's best, in units of the
  feature's standard deviation (which features the model gets wrong);
- confidence: softmax probability of the chosen direction, and the gap head's predicted margin (second
  smallest predicted gap minus the chosen one). AUC for telling bad5 positions apart, and how many bad5
  positions fall in the least confident 5% / 10% / 20%.  -> errors.json"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import torch
L = Path(__file__).resolve().parent; sys.path.insert(0, str(L.parent / 'v51-interpret')); sys.path.insert(0, str(L.parent / 'v49-rules'))
from common import load_net, sample, to_dev, run, gaps, CARD
import features as F49

def feats(x): return F49.features(x[0], [c for c in x[1] if c])
def auc(score, y):
    o = np.argsort(score); r = np.empty(len(o)); r[o] = np.arange(1, len(o) + 1); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

if __name__ == '__main__':
    net = load_net(); a = sample(); n = len(a['board']); g = gaps(a['values'])
    lo, gp = [], []
    for i in range(0, n, 4096):
        x = to_dev(a, slice(i, i + 4096)); l, _, _, _ = run(net, x); lo.append(l); gp.append(net(**x)[1].detach())
    lo = torch.cat(lo); gp = torch.cat(gp).cpu().numpy() / 10
    P = torch.softmax(lo, 1).cpu().numpy(); pick = P.argmax(1); c = g[np.arange(n), pick]; bad = c >= 0.05; best = np.argmin(g, 1)
    out = {'positions': n, 'bad5': float(bad.mean()), 'byPhase': {}}
    for r in range(3, 15):
        m = a['phase'] == r
        if m.sum() >= 300: out['byPhase'][CARD[r]] = {'n': int(m.sum()), 'bad5': float(bad[m].mean())}
    with Pool(12) as p: V = np.stack(p.map(feats, list(zip(a['board'].astype(int).tolist(), a['cards'].tolist())), chunksize=500))
    sd = np.nanstd(V.reshape(-1, V.shape[2]), 0) + 1e-9; idx = np.flatnonzero(bad)
    diff = (V[idx, pick[idx]] - V[idx, best[idx]]) / sd
    out['featureDiff'] = dict(sorted({f: float(np.nanmean(diff[:, k])) for k, f in enumerate(F49.FEATURES)}.items(), key=lambda kv: -abs(kv[1])))
    # same comparison for near-tie choices that differ from the AI (to separate "wrong" from "different")
    alt = np.flatnonzero((c < 0.0025) & (pick != best))
    out['featureDiffNearTies'] = {f: float(np.nanmean((V[alt, pick[alt], k] - V[alt, best[alt], k]) / sd[k])) for k, f in enumerate(F49.FEATURES)}
    ps = np.sort(P, 1); margin_p = ps[:, -1] - ps[:, -2]
    gpm = np.where(a['legal'], gp, np.inf); gs = np.sort(gpm, 1); margin_g = gs[:, 1] - gpm[np.arange(n), pick]
    out['auc'] = {'1 - p(chosen)': auc(1 - P.max(1), bad), 'p margin': auc(-margin_p, bad), 'gap-head margin': auc(-margin_g, bad)}
    cover = {}
    for name, s in (('p(chosen)', -P.max(1)), ('gap-head margin', -margin_g)):
        o = np.argsort(-s)
        cover[name] = {f'{q:.0%}': {'threshold': float(-s[o[int(q * n) - 1]]), 'bad5Caught': float(bad[o[:int(q * n)]].sum() / bad.sum())} for q in (0.05, 0.1, 0.2, 0.3)}
    out['coverage'] = cover
    late = a['phase'] >= 13; out['lateBad5'] = float(bad[late].mean()); out['lateShare'] = float(late.mean())
    (L / 'errors.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n'); print(json.dumps(out, ensure_ascii=False, indent=1))
