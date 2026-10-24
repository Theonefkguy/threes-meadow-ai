"""V57 shared code: the V55 round-21 Shared model, the V51 sample of validation positions, and a forward
pass written out by hand (same arithmetic as models.Shared, attention bias included) that returns the
residual stream after every layer and every head's attention, with mean ablation of heads / MLPs and
zeroing of a layer's attention-bias table.
A "view" is one pass of the network: the board turned so that the judged direction is left. Views are
direction-major: view k*B + i is position i, direction k (0 left, 1 right, 2 up, 3 down).
Tokens: 0 summary, 1-16 cells of the turned board (row-major), 17 preview, 18 counts, 19 cycle.
Run from the repo root:  uv run --project training/v50-transformer python training/v57-interpret-shared/X.py"""
import importlib.util, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
L = Path(__file__).resolve().parent; T = L.parent
sys.path.insert(0, str(T / 'v54-symmetry')); sys.path.insert(0, str(T / 'v50-transformer')); sys.path.insert(0, str(T / 'v49-rules'))
spec = importlib.util.spec_from_file_location('common51', T / 'v51-interpret' / 'common.py'); c51 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c51)
from models import Shared, _tokens
sample, to_dev, gaps, score, CARD, dev, DATA = c51.sample, c51.to_dev, c51.gaps, c51.score, c51.CARD, c51.dev, c51.DATA
MODEL = T / 'v55-shared-dagger' / 'runs' / 'r21' / 'model.pt'
DIR = ['←', '→', '↑', '↓']

def load_net(path=MODEL):
    net = Shared().to(dev); net.load_state_dict(torch.load(path, map_location=dev)); net.eval(); return net

def rotate(net, B):
    """numpy boards [n,16] -> view boards [n,4,16] (view k: direction k turned into left)."""
    return np.asarray(B)[:, net.ROT.cpu().numpy()]

def views(net, x):
    """position inputs -> view inputs (4n, direction-major) and the view legality [4n]."""
    B = x['board'].shape[0]; rep = lambda t: t.repeat(4, *[1] * (t.dim() - 1))
    v = {'board': x['board'][:, net.ROT].transpose(0, 1).reshape(4 * B, 16), 'cards': rep(x['cards']), 'bonus': rep(x['bonus']),
         'counts': rep(x['counts']), 'cycle': rep(x['cycle'])}
    return v, x['legal'].T.reshape(-1)

@torch.no_grad()
def embed(net, v):
    return _tokens(net, v['board'], v['cards'], v['bonus'], v['counts'], v['cycle'], net.cell(v['board'].clamp(max=15)) + net.pos[net.POS])

@torch.no_grad()
def run(net, x, ablate=(), means=None, keep_attn=False, keep_heads=False, hook=None):
    """x: position inputs (to_dev). -> logits [B,4] (illegal masked), view scores [4B] (unmasked),
    gap outputs [B,4], residuals [layers+1] x [4B,20,d], attention [layers] x [4B,H,20,20] (if keep_attn),
    head outputs [layers] x [4B,H,20,dh] (if keep_heads).
    ablate: ('head', layer, h) / ('mlp', layer) replaced by means[...] (from mean_outputs); ('bias', layer) zeroed.
    hook(layer, t) -> t: on the residual stream after each layer (1-based)."""
    v, _ = views(net, x); B = x['board'].shape[0]
    t = embed(net, v); res = [t]; att = []; heads = []
    for li, layer in enumerate(net.layers):
        sa = layer.self_attn; H = sa.num_heads; n, Tk, d = t.shape; dh = d // H
        h = layer.norm1(t)
        q, k, vv = F.linear(h, sa.in_proj_weight, sa.in_proj_bias).split(d, -1)
        q, k, vv = (z.view(n, Tk, H, dh).transpose(1, 2) for z in (q, k, vv))
        m = torch.zeros(H, 20, 20, device=dev)
        if ('bias', li) not in ablate: m[:, 1:17, 1:17] = net.bias[li][:, net.DR, net.DC]
        w = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(dh) + m, -1); o = w @ vv
        for a in ablate:
            if a[0] == 'head' and a[1] == li: o[:, a[2]] = means[('head', li)][a[2]]
        if keep_attn: att.append(w)
        if keep_heads: heads.append(o)
        t = t + sa.out_proj(o.transpose(1, 2).reshape(n, Tk, d))
        mm = layer.linear2(layer.activation(layer.linear1(layer.norm2(t))))
        if ('mlp', li) in ablate: mm = means[('mlp', li)].expand_as(mm)
        t = t + mm
        if hook is not None: t = hook(li + 1, t)
        res.append(t)
    out = net.head(net.norm(t[:, 0])).view(4, B, 2)
    return out[..., 0].T.masked_fill(~x['legal'], -1e9), out[..., 0].reshape(-1), out[..., 1].T, res, att, heads

@torch.no_grad()
def mean_outputs(net, a, n=4096):
    """Mean of every head's output [H,20,dh] and every MLP's output [1,20,d] over the views of the first n positions."""
    x = to_dev(a, slice(0, n)); out = {}
    _, _, _, res, _, heads = run(net, x, keep_heads=True)
    for li, layer in enumerate(net.layers):
        out[('head', li)] = heads[li].mean(0)
        t = res[li] + layer.self_attn.out_proj(heads[li].transpose(1, 2).reshape(*res[li].shape))
        out[('mlp', li)] = layer.linear2(layer.activation(layer.linear1(layer.norm2(t)))).mean(0, keepdim=True)
    return out

def batches(net, a, f, size=4096, **kw):
    """f(run output, slice) over the sample in batches."""
    return [f(run(net, to_dev(a, slice(i, i + size)), **kw), slice(i, min(i + size, len(a['board'])))) for i in range(0, len(a['board']), size)]

def logits_all(net, a, **kw):
    return torch.cat(batches(net, a, lambda r, s: r[0], **kw))

def play(net, n=300, **kw):
    """n whole games with no search (seeds 50900001.., as V51 / V54), run() with ablate / means / hook."""
    from engine import Game, move
    games = [Game(50900001 + i) for i in range(n)]; alive = [g for g in games if not g.over]
    while alive:
        x = {'board': [g.b for g in alive], 'cards': [(g.next['cards'] + [0, 0, 0])[:3] for g in alive], 'bonus': [g.next['bonus'] for g in alive],
             'counts': [g.counts for g in alive], 'cycle': [list(g.cycle) for g in alive], 'legal': [[bool(move(g.b, d)[1]) for d in range(4)] for g in alive]}
        for g, d in zip(alive, run(net, to_dev({k: np.array(v) for k, v in x.items()}), **kw)[0].argmax(1).tolist()): g.step(d)
        alive = [g for g in alive if not g.over and g.turns < 50000]
    top = [max(g.b) for g in games]
    return {'meanScore': float(np.mean([g.score() for g in games])), 'meanMoves': float(np.mean([g.turns for g in games])),
            **{f'reach{c}': float(np.mean([t >= r for t in top])) for c, r in (('1536', 12), ('3072', 13), ('6144', 14))}}

FIG = L / 'fig'
def plt_setup():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif': ['Noto Sans CJK SC', 'Arial Unicode MS'], 'axes.unicode_minus': False,
                         'figure.dpi': 130, 'savefig.bbox': 'tight'})
    FIG.mkdir(exist_ok=True); return plt

if __name__ == '__main__':
    net = load_net(); a = sample(); x = to_dev(a, slice(0, 4096))
    lo, s, gp, *_ = run(net, x); ref, rgap = net(**x)
    print('max |logit diff|', float((lo - ref).abs()[x['legal']].max()), 'max |gap diff|', float((gp - rgap).abs().max()))
    print('near / bad5 on the sample', score(logits_all(net, a), a['values']))
