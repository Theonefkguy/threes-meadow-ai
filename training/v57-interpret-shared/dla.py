"""V57 step 8: exact direct attribution of each view's score (as V52 dla.py).
The final summary residual of a view is the sum of the summary embedding, each head's output (through
out_proj) and each MLP's output at token 0 of every layer, plus the out_proj biases. Given the view's
LayerNorm scale the score is linear in these, so it splits exactly into components.
Over the 30000 validation positions:
- 'decision': contribution to score(chosen) - mean score(other legal directions);
- in bad5 positions: contribution to score(chosen) - score(AI's best).  -> dla.json"""
import json
import numpy as np
import torch
from common import load_net, sample, to_dev, run, gaps, L

net = load_net(); a = sample(); n = len(a['board']); g = gaps(a['values'])
names = ['嵌入'] + [f'层{l + 1}头{h + 1}' for l in range(4) for h in range(4)] + [f'层{l + 1} MLP' for l in range(4)] + ['偏置']
contrib, picks = [], []
with torch.no_grad():
    for i in range(0, n, 2048):
        x = to_dev(a, slice(i, i + 2048)); lo, sc, _, res, _, heads = run(net, x, keep_heads=True); B = lo.shape[0]
        comps = [res[0][:, 0]]; mlps = []
        for l, layer in enumerate(net.layers):
            W = layer.self_attn.out_proj.weight; dh = W.shape[1] // 4
            hc = [heads[l][:, h, 0] @ W[:, h * dh:(h + 1) * dh].T for h in range(4)]; comps += hc
            mlps.append(res[l + 1][:, 0] - res[l][:, 0] - sum(hc) - layer.self_attn.out_proj.bias)
        comps += mlps; comps.append(sum(layer.self_attn.out_proj.bias for layer in net.layers).expand(4 * B, -1))
        r = res[-1][:, 0]; sd = (r.var(1, unbiased=False) + net.norm.eps).sqrt()[:, None]
        wh = net.head.weight[0] * net.norm.weight
        C = torch.stack([((c - c.mean(1, keepdim=True)) / sd) @ wh for c in comps], 1)              # [4B, comps]
        assert torch.allclose(C.sum(1) + net.head.bias[0] + net.head.weight[0] @ net.norm.bias, sc, atol=1e-3)
        contrib.append(C.view(4, B, -1).permute(1, 2, 0).cpu().numpy()); picks.append(lo.argmax(1).cpu().numpy())   # [B, comps, 4]
C = np.concatenate(contrib); pick = np.concatenate(picks); legal = a['legal']; best = g.argmin(1); idx = np.arange(n)
other = np.where(legal[:, None, :], C, np.nan); other[idx, :, pick] = np.nan
decision = C[idx, :, pick] - np.nanmean(other, 2)
bad = g[idx, pick] >= 0.05; kb = np.flatnonzero(bad); wrong = C[kb, :, pick[kb]] - C[kb, :, best[kb]]
out = {'components': names, 'decision': dict(zip(names, np.mean(decision, 0).round(3).tolist())), 'decisionAbs': dict(zip(names, np.mean(np.abs(decision), 0).round(3).tolist())),
       'bad5': int(bad.sum()), 'towardWrong': dict(zip(names, wrong.mean(0).round(3).tolist())), 'totalDecision': float(decision.sum(1).mean()), 'totalWrong': float(wrong.sum(1).mean())}
(L / 'dla.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
for k in names: print(f"{k:8s} decision {out['decision'][k]:+.3f}  |abs| {out['decisionAbs'][k]:.3f}  toward wrong {out['towardWrong'][k]:+.3f}")
print('total', out['totalDecision'], out['totalWrong'])
