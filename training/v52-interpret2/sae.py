"""V52 question 3: sparse autoencoders on the summary token.
Sites: 'L2' = summary token after layer 2 (where V51 found "moves until a larger card"),
'final' = the normalised summary token the output head reads.
Training activations: 600k training AI positions (V46 even seeds + V50 games) and 200k positions from the
student's own games (DAgger round 14). SAE: 128 -> 1024, ReLU, decoder columns of unit norm, loss =
MSE + LAM * L1 (LAM 0.02 after a sweep 0.004 / 0.02 / 0.05 / 0.1, see the report; inputs centred and scaled to unit mean square), Adam 1e-3, 8 epochs, batch 4096.
Evaluation on the 30000 validation positions: fraction of variance unexplained (FVU), active features per
position (L0), dead features, and the model's near / bad5 with the site replaced by its reconstruction.
Labels: Pearson correlation of each feature with a library of known quantities (phase, card counts,
largest card's cell, preview, cycle, empties, mergeable pairs, chain, legal moves, AI value, moves until a
larger card, AI's best / model's chosen direction, the 22 V49 features of each direction).
  -> sae-SITE.pt, sae.json, sae-top.json (top-activating positions of every live feature)"""
import json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
L = Path(__file__).resolve().parent; sys.path.insert(0, str(L.parent / 'v51-interpret')); sys.path.insert(0, str(L.parent / 'v49-rules'))
from common import load_net, sample, to_dev, run, score, gaps, dev, DATA, CARD
from train import arrays, usable
import features as F49
M, LAM = 1024, float(sys.argv[1]) if len(sys.argv) > 1 else 0.02

class SAE(nn.Module):
    def __init__(self, d=128, m=M):
        super().__init__(); self.enc = nn.Linear(d, m); self.dec = nn.Linear(m, d); self.dec.weight.data = nn.functional.normalize(torch.randn(d, m), dim=0)
        self.enc.weight.data = self.dec.weight.data.T.clone(); self.enc.bias.data.zero_()
    def forward(self, x): f = torch.relu(self.enc(x - self.dec.bias)); return self.dec(f), f

@torch.no_grad()
def acts(net, a):
    out = {'L2': [], 'final': []}
    for i in range(0, len(a['board']), 8192):
        _, res, _, _ = run(net, to_dev(a, slice(i, i + 8192))); out['L2'].append(res[2][:, 0]); out['final'].append(net.norm(res[4][:, 0]))
    return {k: torch.cat(v) for k, v in out.items()}

def train_sae(X):
    mu = X.mean(0); sc = ((X - mu) ** 2).sum(1).mean().sqrt() / np.sqrt(X.shape[1]); Z = (X - mu) / sc
    sae = SAE().to(dev); opt = torch.optim.Adam(sae.parameters(), 1e-3); n = len(Z)
    for ep in range(8):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n - 4095, 4096):
            x = Z[perm[i:i + 4096]]; r, f = sae(x); loss = ((r - x) ** 2).mean() + LAM * f.sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad(): sae.dec.weight.data = nn.functional.normalize(sae.dec.weight.data, dim=0)
    return sae, mu, sc

def library(a, V):
    B = a['board'].astype(int); n = len(B); q = {}
    for r in range(9, 15): q[f'阶段={CARD[r]}'] = (a['phase'] == r).astype(float)
    for r in range(0, 15): q[f'{CARD[r]} 的张数'] = (B == r).sum(1)
    mc = B.argmax(1)
    for c in range(16): q[f'最大牌在格{c}'] = (mc == c).astype(float)
    for r in (1, 2, 3): q[f'提示牌含{r}'] = (a['cards'] == r).any(1).astype(float)
    q['奖励牌提示'] = a['bonus'].astype(float)
    for k in range(3): q[f'剩余{k + 1}的张数'] = a['counts'][:, k].astype(float)
    q['周期位置'] = a['cycle'][:, 0].astype(float); q['本周期已出奖励'] = a['cycle'][:, 1].astype(float)
    mergeable = lambda x, y: (x > 0) & (y > 0) & (((x + y) == 3) | ((x >= 3) & (x == y)))
    q['可合并的相邻对数'] = sum(mergeable(B[:, i], B[:, j]) for i, j in F49.ADJ).astype(float)
    q['链长'] = np.array([F49.chain(b) for b in B.tolist()], float); q['合法方向数'] = a['legal'].sum(1).astype(float)
    q['再过几步出现更大的牌(log)'] = np.log1p(np.nan_to_num(a['nextMax'], nan=np.nanmax(a['nextMax'])))
    q['AI估分(log)'] = np.log(np.maximum(np.nanmax(a['values'], 1), 1))
    g = gaps(a['values']); best = g.argmin(1)
    for d, nm in enumerate('←→↑↓'):
        q[f'AI首选{nm}'] = (best == d).astype(float); q[f'{nm}合法'] = a['legal'][:, d].astype(float)
        for k, f in enumerate(F49.FEATURES): q[f'{nm} {f}'] = np.nan_to_num(V[:, d, k], nan=np.nanmin(V[:, :, k]))
    return q

def feats(x): return F49.features(x[0], [c for c in x[1] if c])

if __name__ == '__main__':
    t0 = time.time(); net = load_net(); rng = np.random.default_rng(52)
    d = dict(np.load(DATA / 'positions.npz')); rows = np.flatnonzero(usable(d) & ~d['val']); rows = np.sort(rng.choice(rows, 600000, replace=False))
    e = dict(np.load(DATA / 'dagger' / 'round-14.npz')); r2 = np.flatnonzero(usable(e)); r2 = np.sort(rng.choice(r2, 200000, replace=False))
    A1, A2 = arrays(d, rows), arrays(e, r2); A = {k: np.concatenate([A1[k], A2[k]]) for k in A1}
    Xtr = acts(net, A); v = sample(); Xva = acts(net, v); print(f'activations {time.time() - t0:.0f}s', flush=True)
    with Pool(12) as p: V = np.stack(p.map(feats, list(zip(v['board'].astype(int).tolist(), v['cards'].tolist())), chunksize=500))
    lib = library(v, V); names = list(lib); Q = np.stack([lib[k] for k in names], 1).astype(np.float64)
    Qz = (Q - Q.mean(0)) / (Q.std(0) + 1e-9)
    base = score(run(net, to_dev(v))[0] if False else torch.cat([run(net, to_dev(v, slice(i, i + 8192)))[0] for i in range(0, len(v['board']), 8192)]), v['values'])
    out = {'lambda': LAM, 'baseline': base, 'sites': {}}; top = {}
    for site in ('L2', 'final'):
        sae, mu, sc = train_sae(Xtr[site]); sae.eval()
        torch.save({'state': sae.state_dict(), 'mu': mu, 'sc': sc}, L / f'sae-{site}.pt')
        with torch.no_grad():
            Z = (Xva[site] - mu) / sc; R, F = sae(Z)
            fvu = float(((R - Z) ** 2).sum() / ((Z - Z.mean(0)) ** 2).sum()); l0 = float((F > 0).float().sum(1).mean())
            freq = (F > 0).float().mean(0).cpu().numpy(); Fn = F.cpu().numpy()
            def splice(i0, i1):
                x = to_dev(v, slice(i0, i1)); rec = R[i0:i1] * sc + mu
                if site == 'final':
                    _, res, _, _ = run(net, x); o = net.head(rec); return o[:, :4].masked_fill(~x['legal'], -1e9)
                def hook(l, t):
                    if l == 2: t = t.clone(); t[:, 0] = rec
                    return t
                return run(net, x, hook=hook)[0]
            sp = score(torch.cat([splice(i, min(i + 8192, len(Z))) for i in range(0, len(Z), 8192)]), v['values'])
        alive = np.flatnonzero(freq >= 0.001)
        Fz = (Fn[:, alive] - Fn[:, alive].mean(0)) / (Fn[:, alive].std(0) + 1e-9); C = Fz.T @ Qz / len(Fz)       # [alive, q]
        bi = np.abs(C).argmax(1); br = C[np.arange(len(alive)), bi]
        # sparse features: also the mean z-score of each quantity where the feature is active
        Dz = np.stack([Qz[Fn[:, f] > 0].mean(0) for f in alive])                                                  # [alive, q]
        bj = np.abs(Dz).argmax(1)
        feats_out = [{'id': int(f), 'freq': float(freq[f]), 'label': names[bi[k]], 'r': float(br[k]),
                      'next': [(names[j], float(C[k, j])) for j in np.argsort(-np.abs(C[k]))[1:4]],
                      'whenActive': [(names[j], float(Dz[k, j])) for j in np.argsort(-np.abs(Dz[k]))[:5]]} for k, f in enumerate(alive)]
        feats_out.sort(key=lambda z: -z['freq'])
        hist = {t: int((np.abs(br) >= t).sum()) for t in (0.3, 0.5, 0.7, 0.9)}
        hist.update({f'z>={t}': int((np.abs(Dz).max(1) >= t).sum()) for t in (1, 2, 3)})
        out['sites'][site] = {'fvu': fvu, 'l0': l0, 'alive': int(len(alive)), 'dead': int(M - (freq > 0).sum()), 'spliced': sp, 'labelled': hist, 'features': feats_out}
        print(site, {k: out['sites'][site][k] for k in ('fvu', 'l0', 'alive', 'dead', 'spliced', 'labelled')}, flush=True)
        top[site] = {int(f): [int(i) for i in np.argsort(-Fn[:, f])[:8]] for f in alive}
    (L / 'sae.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n'); (L / 'sae-top.json').write_text(json.dumps(top) + '\n')
    print(f'done {time.time() - t0:.0f}s')
