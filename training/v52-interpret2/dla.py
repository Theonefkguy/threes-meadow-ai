"""V52 question 5: who decides? Exact direct logit attribution on the summary token.
The final summary residual is the sum of: the summary embedding, each head's output (through out_proj)
and each MLP's output at token 0 of every layer. Given each position's LayerNorm scale, the output head is
linear, so every direction logit splits exactly into these components (+ constants).
Reported per component, on the 30000 validation positions:
- 'decision': mean contribution to logit(chosen) - mean logit(other legal directions);
- in bad5 positions (chosen >= 5% worse than the AI's best): mean contribution to
  logit(chosen) - logit(AI's best), i.e. which components push the model to the wrong move.
  -> dla.json"""
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
L = Path(__file__).resolve().parent; sys.path.insert(0, str(L.parent / 'v51-interpret'))
from common import load_net, sample, to_dev, run, gaps, embed

net = load_net(); a = sample(); n = len(a['board']); g = gaps(a['values']); names = ['嵌入'] + [f'层{l + 1}头{h + 1}' for l in range(4) for h in range(4)] + [f'层{l + 1} MLP' for l in range(4)] + ['偏置']
contrib, picks = [], []
with torch.no_grad():
    for i in range(0, n, 4096):
        x = to_dev(a, slice(i, i + 4096)); lo, res, _, heads = run(net, x, keep_heads=True); B = len(lo); comps = [res[0][:, 0]]
        mlps = []
        for l, layer in enumerate(net.enc.layers):
            W = layer.self_attn.out_proj.weight; dh = W.shape[1] // 4
            hc = [heads[l][:, h, 0] @ W[:, h * dh:(h + 1) * dh].T for h in range(4)]; comps += hc
            mlps.append(res[l + 1][:, 0] - res[l][:, 0] - sum(hc) - layer.self_attn.out_proj.bias)
        comps += mlps; comps.append(sum(layer.self_attn.out_proj.bias for layer in net.enc.layers).expand(B, -1))
        r = res[-1][:, 0]; sd = (r.var(1, unbiased=False) + net.norm.eps).sqrt()[:, None]
        Wh = net.head.weight[:4] * net.norm.weight                                    # [4,d]
        C = torch.stack([((c - c.mean(1, keepdim=True)) / sd) @ Wh.T for c in comps], 1)   # [B, comps, 4]
        recon = C.sum(1) + net.head.bias[:4] + net.head.weight[:4] @ net.norm.bias
        assert torch.allclose(recon.masked_fill(~x['legal'], -1e9), lo, atol=1e-3), (recon - lo).abs().max()
        contrib.append(C.cpu().numpy()); picks.append(lo.argmax(1).cpu().numpy())
C = np.concatenate(contrib); pick = np.concatenate(picks); legal = a['legal']; best = g.argmin(1)
idx = np.arange(n); other = np.where(legal[:, None, :], C, np.nan); other[idx, :, pick] = np.nan
decision = C[idx, :, pick] - np.nanmean(other, 2)
bad = g[idx, pick] >= 0.05; wrong = C[bad][np.arange(bad.sum()), :, pick[bad]] - C[bad][np.arange(bad.sum()), :, best[bad]]
out = {'components': names, 'decision': dict(zip(names, np.mean(decision, 0).round(3).tolist())), 'decisionAbs': dict(zip(names, np.mean(np.abs(decision), 0).round(3).tolist())),
       'bad5': int(bad.sum()), 'towardWrong': dict(zip(names, wrong.mean(0).round(3).tolist()))}
(L / 'dla.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
for k in names: print(f"{k:8s} decision {out['decision'][k]:+.3f}  |abs| {out['decisionAbs'][k]:.3f}  toward wrong {out['towardWrong'][k]:+.3f}")
