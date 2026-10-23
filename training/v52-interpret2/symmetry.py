"""V52 question 6: how symmetric is the learned function? Each of the 30000 validation positions is
rotated / mirrored 8 ways (model.SYM), the model's choice mapped back to the original frame.
Reported: how often all 8 agree, the share of the most common choice, near / bad5 of each single frame,
and of the ensemble (log-softmax averaged over the 8 frames); then 300 whole games with the ensemble.
  -> symmetry.json"""
import json, sys
from pathlib import Path
import numpy as np
import torch
L = Path(__file__).resolve().parent; sys.path.insert(0, str(L.parent / 'v51-interpret'))
from common import load_net, sample, to_dev, run, score, SYM, dev
from engine import Game, move

net = load_net(); a = sample(); n = len(a['board'])
CELL = torch.tensor([s[0] for s in SYM], device=dev); DIRS = torch.tensor([s[1] for s in SYM], device=dev)
@torch.no_grad()
def frames(x):
    """log-probs in the original frame for each symmetry: [8, B, 4]"""
    out = []
    for k in range(8):
        y = dict(x); y['board'] = x['board'][:, CELL[k]]; y['legal'] = torch.zeros_like(x['legal']).scatter(1, DIRS[k].expand(len(x['legal']), -1), x['legal'])
        lp = torch.log_softmax(run(net, y)[0], 1)                    # new frame
        out.append(lp[:, DIRS[k]])                                    # back: original direction d -> new DIRS[k][d]
    return torch.stack(out)
F = torch.cat([frames(to_dev(a, slice(i, i + 4096))) for i in range(0, n, 4096)], 1)       # [8,n,4]
picks = F.argmax(2).cpu().numpy(); agree = (picks == picks[0]).all(0)
mode = np.array([np.bincount(picks[:, i], minlength=4).max() for i in range(n)]) / 8
out = {'allAgree': float(agree.mean()), 'meanModeShare': float(mode.mean()),
       'frames': [score(F[k], a['values']) for k in range(8)], 'ensemble': score(F.mean(0), a['values'])}
print(json.dumps(out), flush=True)
def play(ens, n=300):
    games = [Game(50900001 + i) for i in range(n)]; alive = [g for g in games if not g.over]
    while alive:
        x = {'board': [g.b for g in alive], 'cards': [(g.next['cards'] + [0, 0, 0])[:3] for g in alive], 'bonus': [g.next['bonus'] for g in alive],
             'counts': [g.counts for g in alive], 'cycle': [list(g.cycle) for g in alive], 'legal': [[bool(move(g.b, d)[1]) for d in range(4)] for g in alive]}
        x = to_dev({k: np.array(v) for k, v in x.items()})
        ds = (frames(x).mean(0) if ens else run(net, x)[0]).masked_fill(~x['legal'], -1e9).argmax(1).tolist()
        for g, d in zip(alive, ds): g.step(d)
        alive = [g for g in alive if not g.over and g.turns < 50000]
    top = [max(g.b) for g in games]
    return {'meanScore': float(np.mean([g.score() for g in games])), **{f'reach{c}': float(np.mean([t >= r for t in top])) for c, r in (('1536', 12), ('3072', 13), ('6144', 14))}}
out['playSingle'] = play(False); out['playEnsemble'] = play(True); print(out['playSingle'], out['playEnsemble'])
(L / 'symmetry.json').write_text(json.dumps(out, indent=1) + '\n')
