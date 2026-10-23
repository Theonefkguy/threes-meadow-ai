"""V52 question 4: causal tests on the summary token.
(a) The summary token after layer k (k = 1, 2, 3) replaced by its mean over the sample: does the decision
    depend on what the summary token has gathered by then, or is it recomputed from the cell tokens?
(b) Direct logit effect of every live SAE feature at the final site (the output head is linear after the
    LayerNorm): decoder direction x head weights -> change of the 4 direction logits per unit of feature.
    Features labelled with a direction-specific quantity: is their own direction pushed up or down?
(c) Steering at layer 2: add alpha x (the linear probe direction for "log moves until a larger card",
    unit length, alpha in units of the projection's standard deviation) to the summary token; measure
    the chosen moves (V49 features of the chosen direction) and near / bad5.
Measured on the 30000 validation positions, (a) also in 300 whole games with no search.  -> causal.json"""
import json, re, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import torch
L = Path(__file__).resolve().parent; sys.path.insert(0, str(L.parent / 'v51-interpret')); sys.path.insert(0, str(L.parent / 'v49-rules'))
from common import load_net, sample, to_dev, run, score, dev
from engine import Game, move
import features as F49

def feats(x): return F49.features(x[0], [c for c in x[1] if c])
def logits_all(net, a, hook=None):
    return torch.cat([run(net, to_dev(a, slice(i, i + 8192)), hook=hook)[0] for i in range(0, len(a['board']), 8192)])
def play(net, hook=None, n=300):
    games = [Game(50900001 + i) for i in range(n)]; alive = [g for g in games if not g.over]
    while alive:
        x = {'board': [g.b for g in alive], 'cards': [(g.next['cards'] + [0, 0, 0])[:3] for g in alive], 'bonus': [g.next['bonus'] for g in alive],
             'counts': [g.counts for g in alive], 'cycle': [list(g.cycle) for g in alive], 'legal': [[bool(move(g.b, d)[1]) for d in range(4)] for g in alive]}
        for g, d in zip(alive, run(net, to_dev({k: np.array(v) for k, v in x.items()}), hook=hook)[0].argmax(1).tolist()): g.step(d)
        alive = [g for g in alive if not g.over and g.turns < 50000]
    top = [max(g.b) for g in games]
    return {'meanScore': float(np.mean([g.score() for g in games])), 'reach1536': float(np.mean([t >= 12 for t in top])), 'reach3072': float(np.mean([t >= 13 for t in top]))}

if __name__ == '__main__':
    net = load_net(); a = sample(); n = len(a['board']); out = {}
    # residual means of the summary token per layer
    with torch.no_grad():
        _, res, _, _ = run(net, to_dev(a, slice(0, 8192))); mean_sum = [r[:, 0].mean(0) for r in res]
    out['baseline'] = {**score(logits_all(net, a), a['values']), **play(net)}; print('baseline', out['baseline'], flush=True)
    out['summaryAblation'] = {}
    for k in (1, 2, 3):
        def hook(l, t, k=k):
            if l == k: t = t.clone(); t[:, 0] = mean_sum[k]
            return t
        out['summaryAblation'][k] = {**score(logits_all(net, a, hook), a['values']), **play(net, hook)}; print('summary ablated after layer', k, out['summaryAblation'][k], flush=True)
    # (b) direct logit effects of the final-site SAE features
    s = torch.load(L / 'sae-final.pt', map_location=dev); W = s['state']['dec.weight'] * s['sc']               # [128, 1024] residual units
    eff = (net.head.weight[:4] @ W).detach().cpu().numpy().T                                               # [1024, 4]
    sae = json.load(open(L / 'sae.json'))['sites']['final']['features']; arrows = '←→↑↓'; rows = []
    for f in sae:
        nm, z = f['whenActive'][0]; m = re.match(r'^(AI首选)?([←→↑↓])\s*(\S*)', nm)
        if not m: continue
        d = arrows.index(m.group(2)); kind = 'AI首选' if m.group(1) else m.group(3).replace('合法', 'legal')
        own = eff[f['id'], d] - np.delete(eff[f['id']], d).mean()
        rows.append({'id': f['id'], 'kind': kind, 'z': z, 'own': float(own), 'freq': f['freq']})
    summ = {}
    for kind in sorted({r['kind'] for r in rows}):
        rr = [r for r in rows if r['kind'] == kind]; sgn = [np.sign(r['z']) * r['own'] for r in rr]
        summ[kind] = {'features': len(rr), 'ownDirectionPushedUp(sign-adjusted)': float(np.mean(np.array(sgn) > 0)), 'meanEffect': float(np.mean(sgn))}
    out['directLogit'] = summ; print(json.dumps(summ, ensure_ascii=False), flush=True)
    # (c) steering along the layer-2 probe direction for log(moves until a larger card)
    with torch.no_grad():
        X = torch.cat([run(net, to_dev(a, slice(i, i + 8192)))[1][2][:, 0] for i in range(0, n, 8192)]).cpu().numpy()
    y = np.log1p(a['nextMax']); ok = ~np.isnan(y); mu, sd = X[ok].mean(0), X[ok].std(0) + 1e-6
    Z = np.c_[(X[ok] - mu) / sd, np.ones(ok.sum())]; w = np.linalg.solve(Z.T @ Z + 10 * np.eye(Z.shape[1]), Z.T @ y[ok])[:-1] / sd
    u = w / np.linalg.norm(w); proj_sd = float((X @ u).std()); U = torch.tensor(u, dtype=torch.float32, device=dev)
    with Pool(12) as p: V = np.stack(p.map(feats, list(zip(a['board'].astype(int).tolist(), a['cards'].tolist())), chunksize=500))
    base_pick = logits_all(net, a).argmax(1).cpu().numpy(); out['steer'] = {'probeR2train': None, 'projSd': proj_sd, 'alphas': {}}
    for alpha in (-3, -2, -1, 0, 1, 2, 3):
        def hook(l, t, alpha=alpha):
            if l == 2: t = t.clone(); t[:, 0] = t[:, 0] + alpha * proj_sd * U
            return t
        lo = logits_all(net, a, hook); pk = lo.argmax(1).cpu().numpy(); ch = V[np.arange(n), pk]
        r = {**score(lo, a['values']), 'changed': float((pk != base_pick).mean()),
             **{f: float(np.nanmean(ch[:, F49.FEATURES.index(f)])) for f in ('merges', 'mergeMax', 'movedBig', 'bigStill', 'setup', 'empties', 'mono', 'trapped')}}
        out['steer']['alphas'][alpha] = r; print('steer', alpha, {k: round(v, 4) for k, v in r.items()}, flush=True)
    (L / 'causal.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
