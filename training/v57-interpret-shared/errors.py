"""V57 steps 6-7: big mistakes, confidence, and what the per-view score is. 30000 validation positions.
Errors (as V52 errors.py): bad5 (chosen >= 5% worse than the AI's best) by phase; V49 features of the
chosen direction minus the AI's best in bad5 positions and in near ties that differ from the AI (feature
standard deviations); AUC for telling bad5 positions apart from p(chosen), the probability margin and
the gap head's margin; share of bad5 positions in the least confident 5-30%.
Score: each view gives one number without seeing the other directions.
- across all legal views: correlation of the score with log(AI value of the move) and with its gap;
- within positions (>= 2 legal moves): the score minus the position's mean against the gap; Spearman
  between the scores and the AI's values (positions with >= 3 legal moves); share of the score variance
  that lies between positions;
- does the level track the position: the position's mean score vs log(AI's best value).  -> errors.json"""
import json
from multiprocessing import Pool
import numpy as np
import torch
from common import load_net, sample, to_dev, run, gaps, CARD, L
import features as F49

def feats(x): return F49.features(x[0], [c for c in x[1] if c])
def auc(s, y):
    o = np.argsort(s); r = np.empty(len(o)); r[o] = np.arange(1, len(o) + 1); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def rank(x): r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); return r

if __name__ == '__main__':
    net = load_net(); a = sample(); n = len(a['board']); g = gaps(a['values']); legal = a['legal']
    lo, gp = [], []
    for i in range(0, n, 4096):
        l, _, gg, _, _, _ = run(net, to_dev(a, slice(i, i + 4096))); lo.append(l); gp.append(gg)
    lo = torch.cat(lo); S = lo.cpu().numpy(); gp = torch.cat(gp).cpu().numpy() / 10
    P = torch.softmax(lo, 1).cpu().numpy(); pick = P.argmax(1); c = g[np.arange(n), pick]; bad = c >= 0.05; best = np.argmin(g, 1)
    out = {'positions': n, 'near': float((c < 0.0025).mean()), 'bad5': float(bad.mean()), 'byPhase': {}}
    for r in range(3, 15):
        m = a['phase'] == r
        if m.sum() >= 300: out['byPhase'][CARD[r]] = {'n': int(m.sum()), 'bad5': float(bad[m].mean())}
    with Pool(12) as p: V = np.stack(p.map(feats, list(zip(a['board'].astype(int).tolist(), a['cards'].tolist())), chunksize=500))
    sd = np.nanstd(V.reshape(-1, V.shape[2]), 0) + 1e-9; idx = np.flatnonzero(bad)
    out['featureDiff'] = dict(sorted({f: float(np.nanmean((V[idx, pick[idx], k] - V[idx, best[idx], k]) / sd[k])) for k, f in enumerate(F49.FEATURES)}.items(), key=lambda kv: -abs(kv[1])))
    alt = np.flatnonzero((c < 0.0025) & (pick != best))
    out['featureDiffNearTies'] = {f: float(np.nanmean((V[alt, pick[alt], k] - V[alt, best[alt], k]) / sd[k])) for k, f in enumerate(F49.FEATURES)}
    ps = np.sort(P, 1); margin_p = ps[:, -1] - ps[:, -2]
    gpm = np.where(legal, gp, np.inf); gs = np.sort(gpm, 1); margin_g = gs[:, 1] - gpm[np.arange(n), pick]
    out['auc'] = {'1 - p(chosen)': auc(1 - P.max(1), bad), 'p margin': auc(-margin_p, bad), 'gap-head margin': auc(-margin_g, bad)}
    out['coverage'] = {}
    for name, s in (('p(chosen)', -P.max(1)), ('gap-head margin', -margin_g)):
        o = np.argsort(-s); out['coverage'][name] = {f'{q:.0%}': float(bad[o[:int(q * n)]].sum() / bad.sum()) for q in (0.05, 0.1, 0.2, 0.3)}
    # the per-view score
    v = a['values'].astype(float); lv = legal
    with np.errstate(divide='ignore', invalid='ignore'): logv = np.log(np.maximum(v, 1))
    s_l, lv_l, g_l = S[lv], logv[lv], np.minimum(g[lv], 0.1)
    mean_s = np.array([S[i, lv[i]].mean() for i in range(n)]); k2 = lv.sum(1) >= 2
    within = (S - mean_s[:, None])[lv & k2[:, None]]; gw = np.minimum(g, 0.1)[lv & k2[:, None]]
    sp = [np.corrcoef(rank(S[i, lv[i]]), rank(v[i, lv[i]]))[0, 1] for i in np.flatnonzero(lv.sum(1) >= 3)]
    top_ok = np.mean([S[i, lv[i]].argmax() == v[i, lv[i]].argmax() for i in np.flatnonzero(k2)])
    tot = s_l.var(); betw = np.repeat(mean_s, lv.sum(1))
    out['score'] = {'corrLogValue(all views)': float(np.corrcoef(s_l, lv_l)[0, 1]), 'corrGap(all views)': float(np.corrcoef(s_l, g_l)[0, 1]),
                    'corrGapWithinPosition': float(np.corrcoef(within, gw)[0, 1]), 'spearmanWithinPosition(>=3 legal)': float(np.nanmean(sp)),
                    'betweenPositionShareOfVariance': float(betw.var() / tot),
                    'corrPositionMeanVsLogBest': float(np.corrcoef(mean_s, np.log(np.maximum(np.nanmax(v, 1), 1)))[0, 1]),
                    'scoreRange': [float(np.percentile(s_l, 1)), float(np.percentile(s_l, 99))]}
    late = a['phase'] >= 13; out['lateBad5'] = float(bad[late].mean()); out['lateShare'] = float(late.mean())
    (L / 'errors.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n'); print(json.dumps(out, ensure_ascii=False, indent=1))
