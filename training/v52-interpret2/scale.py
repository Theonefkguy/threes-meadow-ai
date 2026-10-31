"""V52 question 5b: causal check of the DLA finding. The outputs of layer-4 heads 3 and 4 at the summary
token are multiplied by s (only the summary token matters after layer 4). Measured on the 30000 validation
positions (near, bad5, bad5 in the 3072+ phase) and, for every s, in 300 whole games with no search
(seeds 50900001..; the V51 baseline is 54% reaching 3072).  -> scale.json"""
import json, sys
from pathlib import Path
import numpy as np
import torch
L = Path(__file__).resolve().parent; sys.path.insert(0, str(L.parent / 'v51-interpret'))
from common import load_net, sample, to_dev, run, score, gaps
from engine import Game, move

net = load_net(); a = sample(); g = gaps(a['values'])
def logits(x, s):
    lo, res, _, heads = run(net, x, keep_heads=True); W = net.enc.layers[3].self_attn.out_proj.weight; dh = W.shape[1] // 4
    r = res[-1][:, 0].clone()
    for h, sh in s.items(): r = r - (1 - sh) * (heads[3][:, h, 0] @ W[:, h * dh:(h + 1) * dh].T)
    return net.head(net.norm(r))[:, :4].masked_fill(~x['legal'], -1e9)
def play(s, n=300):
    games = [Game(50900001 + i) for i in range(n)]; alive = [gm for gm in games if not gm.over]
    while alive:
        x = {'board': [gm.b for gm in alive], 'cards': [(gm.next['cards'] + [0, 0, 0])[:3] for gm in alive], 'bonus': [gm.next['bonus'] for gm in alive],
             'counts': [gm.counts for gm in alive], 'cycle': [list(gm.cycle) for gm in alive], 'legal': [[bool(move(gm.b, d)[1]) for d in range(4)] for gm in alive]}
        with torch.no_grad(): ds = logits(to_dev({k: np.array(v) for k, v in x.items()}), s).argmax(1).tolist()
        for gm, d in zip(alive, ds): gm.step(d)
        alive = [gm for gm in alive if not gm.over and gm.turns < 50000]
    top = [max(gm.b) for gm in games]
    return {'meanScore': float(np.mean([gm.score() for gm in games])), 'reach3072': float(np.mean([t >= 13 for t in top])), 'reach6144': float(np.mean([t >= 14 for t in top]))}
out = {}
for name, s in [('原样', {}), ('头3、头4 ×0.5', {2: 0.5, 3: 0.5}), ('头3、头4 ×0.75', {2: 0.75, 3: 0.75}), ('头3、头4 ×1.25', {2: 1.25, 3: 1.25}),
                ('头4 ×0.75', {3: 0.75}), ('头3 ×0.75', {2: 0.75})]:
    with torch.no_grad(): lo = torch.cat([logits(to_dev(a, slice(i, i + 8192)), s) for i in range(0, len(g), 8192)])
    pk = lo.argmax(1).cpu().numpy(); c = g[np.arange(len(pk)), pk]; late = a['phase'] >= 13
    out[name] = {**score(lo, a['values']), 'bad5late': float((c[late] >= 0.05).mean()), **play(s)}
    print(name, {k: round(v, 4) for k, v in out[name].items()}, flush=True)
(L / 'scale.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
