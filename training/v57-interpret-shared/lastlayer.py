"""V57 note: layer 4 computed for the summary token only (the cell tokens' layer-4 outputs are never read,
so their queries and MLP can be skipped; keys / values of all tokens are still needed). Checks the scores
against the model and times both on one CPU thread, batch 1 and 64.
  TORCH_DEVICE=cpu uv run --project training/v50-transformer python training/v57-interpret-shared/lastlayer.py"""
import math, sys, time
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import torch, torch.nn.functional as F
from common import load_net, sample, to_dev, views, embed
torch.set_num_threads(1)
net = load_net(); a = sample(); x = to_dev(a, slice(0, 2048))
@torch.no_grad()
def fast(x):
    v, _ = views(net, x); B = x['board'].shape[0]; t = embed(net, v)
    for li, layer in enumerate(net.layers[:-1]):
        m = torch.zeros(net.H, 20, 20, device=t.device); m[:, 1:17, 1:17] = net.bias[li][:, net.DR, net.DC]
        t = layer(t, src_mask=m.repeat(t.shape[0], 1, 1))
    L = net.layers[-1]; sa = L.self_attn; n, T, d = t.shape; H = sa.num_heads; dh = d // H
    h = L.norm1(t); W, b = sa.in_proj_weight, sa.in_proj_bias
    q = F.linear(h[:, :1], W[:d], b[:d]); kv = F.linear(h, W[d:], b[d:]); k, vv = kv.split(d, -1)
    q = q.view(n, 1, H, dh).transpose(1, 2); k = k.view(n, T, H, dh).transpose(1, 2); vv = vv.view(n, T, H, dh).transpose(1, 2)
    o = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(dh), -1) @ vv
    s = t[:, 0] + sa.out_proj(o.transpose(1, 2).reshape(n, d))
    s = s + L.linear2(L.activation(L.linear1(L.norm2(s))))
    return net.head(net.norm(s)).view(4, B, 2)[..., 0].T.masked_fill(~x['legal'], -1e9)
with torch.no_grad():
    ref = net(**x)[0]; f = fast(x)
    print('max diff', float((ref - f).abs()[x['legal']].max()), 'same choice', float((ref.argmax(1) == f.argmax(1)).float().mean()))
    for name, fn in (('full', lambda y: net(**y)), ('layer 4 summary only', fast)):
        for bs in (1, 64):
            xs = [to_dev(a, slice(i, i + bs)) for i in range(0, 200 * bs, bs)][:200]
            fn(xs[0]); t0 = time.perf_counter()
            for y in xs: fn(y)
            print(f'{name:22s} batch {bs:3d}: {(time.perf_counter() - t0) / len(xs) / bs * 1000:.3f} ms/position')
