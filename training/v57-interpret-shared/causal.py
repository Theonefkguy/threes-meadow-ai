"""V57 step 10: causal tests on the summary token (as V52 causal.py (a), (b)).
(a) The summary token of every view after layer k (k = 1, 2, 3) replaced by its mean over the views of
    8192 positions: near / bad5 on the 30000 validation positions and 300 whole games with no search.
(b) Direct effect of every live final-site SAE feature on the view's score (the head is linear after the
    LayerNorm): decoder direction x score weights. Features grouped by the quantity that is most extreme
    when they are active (sae.json 'whenActive'); for each group the share whose effect pushes the score
    in the direction "high quantity -> higher score".  -> causal.json"""
import json
import numpy as np
import torch
from common import load_net, sample, to_dev, run, logits_all, play, score, dev, L

net = load_net(); a = sample(); out = {}
with torch.no_grad():
    _, _, _, res, _, _ = run(net, to_dev(a, slice(0, 8192))); mean_sum = [r[:, 0].mean(0) for r in res]
out['baseline'] = {**score(logits_all(net, a), a['values']), **play(net)}; print('baseline', out['baseline'], flush=True)
out['summaryAblation'] = {}
for k in (1, 2, 3):
    def hook(l, t, k=k):
        if l == k: t = t.clone(); t[:, 0] = mean_sum[k]
        return t
    out['summaryAblation'][k] = {**score(logits_all(net, a, hook=hook), a['values']), **play(net, hook=hook)}; print('summary ablated after layer', k, out['summaryAblation'][k], flush=True)
s = torch.load(L / 'sae-final.pt', map_location=dev); eff = (net.head.weight[0] @ (s['state']['dec.weight'] * s['sc'])).detach().cpu().numpy()   # [1024]
feats = json.load(open(L / 'sae.json'))['sites']['final']['features']; groups = {}
for f in feats:
    nm, z = f['whenActive'][0]; groups.setdefault(nm, []).append(float(np.sign(z) * eff[f['id']]))
out['directEffect'] = {k: {'features': len(v), 'highQuantityRaisesScore': float(np.mean(np.array(v) > 0)), 'meanSignedEffect': float(np.mean(v))}
                       for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))}
for k, v in list(out['directEffect'].items())[:25]: print(k, v)
(L / 'causal.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
