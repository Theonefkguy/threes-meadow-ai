"""V51 shared code: the V50 DAgger-14 transformer, a sample of validation positions, and a forward pass
written out by hand (same arithmetic as model.Net) that returns the residual stream after every layer
and every head's attention, and can replace any head's or MLP's output by its mean (mean ablation).
Tokens: 0 summary, 1-16 cells (row-major, cell i = token i + 1), 17 preview, 18 counts, 19 cycle.
Run from the repo root:  uv run --project training/v50-transformer python training/v51-interpret/X.py"""
import math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
L = Path(__file__).resolve().parent; V50 = L.parent / 'v50-transformer'
sys.path.insert(0, str(V50))
from model import Net, SYM
from train import arrays, usable, DATA
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
MODEL = V50 / 'runs' / 'dag14' / 'model.pt'
TOK = ['summary'] + [f'c{i}' for i in range(16)] + ['preview', 'counts', 'cycle']

def load_net(path=MODEL):
    net = Net().to(dev); net.load_state_dict(torch.load(path, map_location=dev)); net.eval(); return net

def sample(n=30000, seed=51):
    """n validation positions (V46 odd seeds), plus 'nextMax': moves until the AI's game first shows a
    larger top card (NaN if never), 'phase' (top rank) and 'seed'/'move'."""
    d = dict(np.load(DATA / 'positions.npz')); ok = usable(d) & d['val']
    top = d['board'].max(1); nxt = np.full(len(top), np.nan)
    order = np.lexsort((d['move'], d['seed']))
    for s in np.unique(d['seed'][d['val']]):
        idx = order[d['seed'][order] == s]; t = top[idx]; nm = np.nan
        for j in range(len(idx) - 1, -1, -1):
            if j + 1 < len(idx) and t[j + 1] > t[j]: nm = d['move'][idx[j + 1]]
            nxt[idx[j]] = nm - d['move'][idx[j]] if not np.isnan(nm) else np.nan
    rows = np.flatnonzero(ok); rows = np.sort(np.random.default_rng(seed).choice(rows, min(n, len(rows)), replace=False))
    a = arrays(d, rows)
    a.update(values=d['values'][rows], seed=d['seed'][rows], move=d['move'][rows], nextMax=nxt[rows], phase=top[rows])
    return a

def to_dev(a, idx=slice(None)):
    t = lambda k, dt=torch.long: torch.as_tensor(np.ascontiguousarray(a[k][idx])).to(dev, dt)
    return {'board': t('board'), 'cards': t('cards'), 'bonus': t('bonus', torch.bool), 'counts': t('counts'), 'cycle': t('cycle'), 'legal': t('legal', torch.bool)}

@torch.no_grad()
def embed(net, x):
    B = x['board'].shape[0]
    c = net.cell(x['board'].clamp(max=15)) + net.pos
    pv = torch.zeros(B, 17, device=dev).scatter_(1, x['cards'], 1.0); pv[:, 0] = 0; pv[:, 16] = x['bonus'].float()
    cn = F.one_hot(x['counts'].clamp(0, 4), 5).flatten(1).float()
    cy = torch.cat([F.one_hot(x['cycle'][:, 0].clamp(0, 20), 21), x['cycle'][:, 1:2].clamp(0, 1), F.one_hot(x['cycle'][:, 2].clamp(0, 21), 22)], 1).float()
    return torch.cat([net.summary.expand(B, -1, -1), c, net.preview(pv)[:, None], net.counts(cn)[:, None], net.cycle(cy)[:, None]], 1)

@torch.no_grad()
def run(net, x, ablate=(), means=None, keep_attn=False, keep_heads=False, hook=None):
    """-> logits [B,4], residuals [layers+1] x [B,20,d], attention [layers] x [B,H,20,20] (if keep_attn),
    head outputs [layers] x [B,H,20,dh] (if keep_heads).
    ablate: ('head', layer, h) / ('mlp', layer) entries, replaced by means[...] (from mean_outputs).
    hook(layer, t) -> t: called on the residual stream after each layer (1-based layer number)."""
    t = embed(net, x); res = [t]; att = []; heads = []
    for li, layer in enumerate(net.enc.layers):
        sa = layer.self_attn; H = sa.num_heads; d = t.shape[-1]; dh = d // H; B, T, _ = t.shape
        h = layer.norm1(t)
        q, k, v = F.linear(h, sa.in_proj_weight, sa.in_proj_bias).split(d, -1)
        q, k, v = (z.view(B, T, H, dh).transpose(1, 2) for z in (q, k, v))
        w = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(dh), -1); o = w @ v          # [B,H,T,dh]
        for a in ablate:
            if a[0] == 'head' and a[1] == li: o[:, a[2]] = means[('head', li)][a[2]]
        if keep_attn: att.append(w)
        if keep_heads: heads.append(o)
        t = t + sa.out_proj(o.transpose(1, 2).reshape(B, T, d))
        m = layer.linear2(layer.activation(layer.linear1(layer.norm2(t))))
        if ('mlp', li) in ablate: m = means[('mlp', li)].expand_as(m)
        t = t + m
        if hook is not None: t = hook(li + 1, t)
        res.append(t)
    out = net.head(net.norm(t[:, 0]))
    return out[:, :4].masked_fill(~x['legal'], -1e9), res, att, heads

@torch.no_grad()
def mean_outputs(net, a, n=8192):
    """Mean of every head's output [H,20,dh] and every MLP's output [20,d] over the first n positions."""
    x = to_dev(a, slice(0, n)); out = {}
    _, res, _, heads = run(net, x, keep_heads=True)
    for li, layer in enumerate(net.enc.layers):
        out[('head', li)] = heads[li].mean(0)
        m = layer.linear2(layer.activation(layer.linear1(layer.norm2(res[li] + layer.self_attn.out_proj(heads[li].transpose(1, 2).reshape(*res[li].shape))))))
        out[('mlp', li)] = m.mean(0, keepdim=True)
    return out

def gaps(values):
    v = values.astype(np.float64); legal = ~np.isnan(v)
    with np.errstate(invalid='ignore', divide='ignore'):
        top = np.nanmax(v, 1); g = (top[:, None] - v) / np.abs(top)[:, None]
    return np.where(legal, g, np.inf)

def score(logits, values):
    """near (within 0.25% of the AI's best) and bad5 (>= 5% worse) of the argmax choices."""
    g = gaps(values); p = logits.argmax(1).cpu().numpy(); c = g[np.arange(len(p)), p]
    return {'near': float((c < 0.0025).mean()), 'bad5': float((c >= 0.05).mean())}

CARD = ['空', '1', '2', '3'] + [str(3 * 2 ** (r - 3)) for r in range(4, 16)]
FIG = L / 'fig'
def plt_setup():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif': ['Noto Sans CJK SC', 'Arial Unicode MS'], 'axes.unicode_minus': False,
                         'figure.dpi': 130, 'savefig.bbox': 'tight'})
    FIG.mkdir(exist_ok=True); return plt
