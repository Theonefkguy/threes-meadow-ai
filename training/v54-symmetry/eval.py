"""V54 evaluation of the from-scratch models (runs/ARCH/model.pt) and V50 dag14 as a reference:
near / bad5 on all validation positions; on 30000 of them the share where all 8 rotations / mirrors pick
the same direction; 300 whole games (seeds 50900001.., no search); for base and dag14 also the 8-frame
ensemble (log-softmax averaged).  -> eval.json
  uv run --project training/v50-transformer python training/v54-symmetry/eval.py [ARCH ...]"""
import json, sys
from pathlib import Path
import numpy as np
import torch
L = Path(__file__).resolve().parent; sys.path.insert(0, str(L)); sys.path.insert(0, str(L.parent / 'v50-transformer'))
from models import Net, Canon, Shared, SYM
from train import arrays, on_dev, batch, usable, evaluate, DATA, dev
from engine import Game, move
CELL = torch.tensor([s[0] for s in SYM], device=dev); DIRS = torch.tensor([s[1] for s in SYM], device=dev)
ARCH = {'base': Net, 'canon': Canon, 'shared': Shared, 'dag14': Net}
KEYS = ('board', 'cards', 'bonus', 'counts', 'cycle', 'legal')

@torch.no_grad()
def frames(net, x):
    """log-probs in the original frame for each symmetry [8,B,4]"""
    out = []
    for k in range(8):
        y = dict(x); y['board'] = x['board'][:, CELL[k]]; y['legal'] = torch.zeros_like(x['legal']).scatter(1, DIRS[k].expand(len(x['legal']), -1), x['legal'])
        out.append(torch.log_softmax(net(*(y[k2] for k2 in KEYS))[0], 1)[:, DIRS[k]])
    return torch.stack(out)

@torch.no_grad()
def play(net, ens, n=300):
    games = [Game(50900001 + i) for i in range(n)]; alive = list(games)
    while alive:
        x = {'board': [g.b for g in alive], 'cards': [(g.next['cards'] + [0, 0, 0])[:3] for g in alive], 'bonus': [g.next['bonus'] for g in alive],
             'counts': [g.counts for g in alive], 'cycle': [list(g.cycle) for g in alive], 'legal': [[bool(move(g.b, d)[1]) for d in range(4)] for g in alive]}
        x = batch(on_dev({k: np.array(v) for k, v in x.items()}))
        lo = frames(net, x).mean(0).masked_fill(~x['legal'], -1e9) if ens else net(*(x[k] for k in KEYS))[0]
        for g, d in zip(alive, lo.argmax(1).tolist()): g.step(d)
        alive = [g for g in alive if not g.over and g.turns < 50000]
    top = [max(g.b) for g in games]
    return {'meanScore': float(np.mean([g.score() for g in games])), 'meanMoves': float(np.mean([g.turns for g in games])),
            **{f'reach{c}': float(np.mean([t >= r for t in top])) for c, r in (('1536', 12), ('3072', 13), ('6144', 14), ('12288', 15))}}

if __name__ == '__main__':
    archs = sys.argv[1:] or ['dag14', 'base', 'canon', 'shared']
    d = dict(np.load(DATA / 'positions.npz')); va = on_dev(arrays(d, usable(d) & d['val']))
    sub = batch(va, torch.as_tensor(np.sort(np.random.default_rng(54).choice(len(va['board']), 30000, replace=False)), device=dev))
    f = L / 'eval.json'; out = json.loads(f.read_text()) if f.exists() else {}
    for a in archs:
        net = ARCH[a]().to(dev); path = L.parent / 'v50-transformer' / 'runs' / 'dag14' / 'model.pt' if a == 'dag14' else L / 'runs' / a / 'model.pt'
        net.load_state_dict(torch.load(path, map_location=dev)); net.eval()
        F = torch.cat([frames(net, {k: v[i:i + 4096] for k, v in sub.items()}) for i in range(0, 30000, 4096)], 1); p = F.argmax(2)
        r = {'val': evaluate(net, va), 'allAgree': float((p == p[0]).all(0).float().mean()), 'play': play(net, False)}
        if a in ('base', 'dag14'): r['playEnsemble'] = play(net, True)
        out[a] = r; print(a, json.dumps(r), flush=True); f.write_text(json.dumps(out, indent=1) + '\n')
