"""V57 step 9: sparse autoencoders on the summary token of each view (as V52 sae.py).
Sites: 'L2' = summary token after layer 2, 'final' = the normalised summary token the score head reads.
Training activations: the legal views of 600k training AI positions (V46 even seeds + V50 games) and of
200k positions of the student's own games (V55 DAgger round 21). SAE 128 -> 1024, ReLU, unit-norm decoder
columns, MSE + 0.02 x L1 on inputs centred and scaled to unit mean square, Adam 1e-3, 8 epochs, batch 4096.
Evaluation on the legal views of the 30000 validation positions: FVU, L0, dead features, near / bad5 with
the site replaced by its reconstruction (all views). Labels: correlation of each feature with quantities of
the view (this move's 22 V49 features, the AI's value / gap / best for this move, where the largest card
is on the turned board) and of the position (phase, card counts, preview, cycle, empties, pairs, chain,
moves until a larger card, AI's best value).  -> sae-SITE.pt, sae.json"""
import json, time
from multiprocessing import Pool
import numpy as np
import torch
import torch.nn as nn
from common import load_net, sample, to_dev, run, rotate, score, gaps, dev, DATA, CARD, L
from train import arrays, usable
import features as F49
M, LAM = 1024, 0.02

class SAE(nn.Module):
    def __init__(self, d=128, m=M):
        super().__init__(); self.enc = nn.Linear(d, m); self.dec = nn.Linear(m, d); self.dec.weight.data = nn.functional.normalize(torch.randn(d, m), dim=0)
        self.enc.weight.data = self.dec.weight.data.T.clone(); self.enc.bias.data.zero_()
    def forward(self, x): f = torch.relu(self.enc(x - self.dec.bias)); return self.dec(f), f

@torch.no_grad()
def acts(net, a, legal_only=True):
    """site -> [views, d], position-major (position i, direction k at i * 4 + k), legal views only if asked."""
    out = {'L2': [], 'final': []}
    for i in range(0, len(a['board']), 4096):
        x = to_dev(a, slice(i, i + 4096)); B = x['board'].shape[0]; _, _, _, res, _, _ = run(net, x)
        pm = lambda t: t.view(4, B, -1).transpose(0, 1).reshape(4 * B, -1)
        l2, fin = pm(res[2][:, 0]), pm(net.norm(res[4][:, 0]))
        if legal_only: m = x['legal'].reshape(-1); l2, fin = l2[m], fin[m]
        out['L2'].append(l2); out['final'].append(fin)
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

def feats(x): return F49.features(x[0], [c for c in x[1] if c])
def library(net, a, V):
    """quantities per legal view (position-major order, as acts)."""
    B = a['board'].astype(int); lv = a['legal'].reshape(-1); per_pos = {}
    for r in range(9, 15): per_pos[f'阶段={CARD[r]}'] = (a['phase'] == r).astype(float)
    for r in range(0, 15): per_pos[f'{CARD[r]} 的张数'] = (B == r).sum(1)
    for r in (1, 2, 3): per_pos[f'提示牌含{r}'] = (a['cards'] == r).any(1).astype(float)
    per_pos['奖励牌提示'] = a['bonus'].astype(float)
    for k in range(3): per_pos[f'剩余{k + 1}的张数'] = a['counts'][:, k].astype(float)
    per_pos['周期位置'] = a['cycle'][:, 0].astype(float); per_pos['本周期已出奖励'] = a['cycle'][:, 1].astype(float)
    mergeable = lambda x, y: (x > 0) & (y > 0) & (((x + y) == 3) | ((x >= 3) & (x == y)))
    per_pos['可合并的相邻对数'] = sum(mergeable(B[:, i], B[:, j]) for i, j in F49.ADJ).astype(float)
    per_pos['链长'] = np.array([F49.chain(b) for b in B.tolist()], float); per_pos['合法方向数'] = a['legal'].sum(1).astype(float)
    per_pos['再过几步出现更大的牌(log)'] = np.log1p(np.nan_to_num(a['nextMax'], nan=np.nanmax(a['nextMax'])))
    per_pos['AI估分(log)'] = np.log(np.maximum(np.nanmax(a['values'], 1), 1))
    q = {k: np.repeat(v.astype(float), 4)[lv] for k, v in per_pos.items()}
    g = gaps(a['values']); v = a['values'].astype(float)
    with np.errstate(divide='ignore', invalid='ignore'): logv = np.log(np.maximum(v, 1))
    q['这一步：AI估分(log)'] = logv.reshape(-1)[lv]; q['这一步：比首选差(≤10%)'] = np.minimum(g, 0.1).reshape(-1)[lv]; q['这一步：AI首选'] = (g == 0).reshape(-1)[lv].astype(float)
    VB = rotate(net, B).reshape(-1, 16)[lv]; mc = VB.argmax(1)
    for c in range(16): q[f'视角里最大牌在格{c}'] = (mc == c).astype(float)
    for c in range(4): q[f'视角里最大牌在第{c + 1}列'] = (mc % 4 == c).astype(float)
    q['视角里最大牌在边行'] = np.isin(mc // 4, [0, 3]).astype(float)
    for k, f in enumerate(F49.FEATURES): q[f'这一步 {f}'] = np.nan_to_num(V[:, :, k].reshape(-1)[lv], nan=np.nanmin(V[:, :, k]))
    return q

if __name__ == '__main__':
    t0 = time.time(); net = load_net(); rng = np.random.default_rng(57)
    d = dict(np.load(DATA / 'positions.npz')); rows = np.flatnonzero(usable(d) & ~d['val']); rows = np.sort(rng.choice(rows, 600000, replace=False))
    e = dict(np.load(DATA / 'dagger' / 'round-21.npz')); r2 = np.flatnonzero(usable(e)); r2 = np.sort(rng.choice(r2, 200000, replace=False))
    A1, A2 = arrays(d, rows), arrays(e, r2); A = {k: np.concatenate([A1[k], A2[k]]) for k in A1}
    Xtr = acts(net, A); v = sample(); Xva = acts(net, v); print(f'activations {len(Xtr["L2"])} train views, {time.time() - t0:.0f}s', flush=True)
    with Pool(12) as p: V = np.stack(p.map(feats, list(zip(v['board'].astype(int).tolist(), v['cards'].tolist())), chunksize=500))
    lib = library(net, v, V); names = list(lib); Q = np.stack([lib[k] for k in names], 1).astype(np.float64); Qz = (Q - Q.mean(0)) / (Q.std(0) + 1e-9)
    lg = lambda **kw: torch.cat([run(net, to_dev(v, slice(i, i + 4096)), **kw)[0] for i in range(0, len(v['board']), 4096)])
    out = {'lambda': LAM, 'trainViews': len(Xtr['L2']), 'baseline': score(lg(), v['values']), 'sites': {}}
    for site in ('L2', 'final'):
        sae, mu, sc = train_sae(Xtr[site]); sae.eval()
        torch.save({'state': sae.state_dict(), 'mu': mu, 'sc': sc}, L / f'sae-{site}.pt')
        with torch.no_grad():
            Z = (Xva[site] - mu) / sc; R, F = sae(Z)
            fvu = float(((R - Z) ** 2).sum() / ((Z - Z.mean(0)) ** 2).sum()); l0 = float((F > 0).float().sum(1).mean())
            freq = (F > 0).float().mean(0).cpu().numpy(); Fn = F.cpu().numpy()
            if site == 'final':
                def splice(x):
                    B = x['board'].shape[0]; _, _, _, res, _, _ = run(net, x); f = net.norm(res[4][:, 0])
                    s = net.head(sae((f - mu) / sc)[0] * sc + mu)[:, 0].view(4, B).T; return s.masked_fill(~x['legal'], -1e9)
            else:
                def splice(x):
                    def hook(l, t):
                        if l == 2: t = t.clone(); t[:, 0] = sae((t[:, 0] - mu) / sc)[0] * sc + mu
                        return t
                    return run(net, x, hook=hook)[0]
            sp = score(torch.cat([splice(to_dev(v, slice(i, i + 4096))) for i in range(0, len(v['board']), 4096)]), v['values'])
        alive = np.flatnonzero(freq >= 0.001)
        Fz = (Fn[:, alive] - Fn[:, alive].mean(0)) / (Fn[:, alive].std(0) + 1e-9); C = Fz.T @ Qz / len(Fz)
        bi = np.abs(C).argmax(1); br = C[np.arange(len(alive)), bi]
        Dz = np.stack([Qz[Fn[:, f] > 0].mean(0) for f in alive])
        feats_out = [{'id': int(f), 'freq': float(freq[f]), 'label': names[bi[k]], 'r': float(br[k]),
                      'next': [(names[j], float(C[k, j])) for j in np.argsort(-np.abs(C[k]))[1:4]],
                      'whenActive': [(names[j], float(Dz[k, j])) for j in np.argsort(-np.abs(Dz[k]))[:5]]} for k, f in enumerate(alive)]
        feats_out.sort(key=lambda z: -z['freq'])
        hist = {t: int((np.abs(br) >= t).sum()) for t in (0.3, 0.5, 0.7, 0.9)}; hist.update({f'z>={t}': int((np.abs(Dz).max(1) >= t).sum()) for t in (1, 2, 3)})
        out['sites'][site] = {'fvu': fvu, 'l0': l0, 'alive': int(len(alive)), 'dead': int(M - (freq > 0).sum()), 'spliced': sp, 'labelled': hist, 'features': feats_out}
        print(site, {k: out['sites'][site][k] for k in ('fvu', 'l0', 'alive', 'dead', 'spliced', 'labelled')}, flush=True)
    (L / 'sae.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n'); print(f'done {time.time() - t0:.0f}s')
