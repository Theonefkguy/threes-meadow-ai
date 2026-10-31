"""V56 evaluation of runs/NAME (s128, s256): validation and training-set numbers (300k training rows of the
V46 AI games and of DAgger rounds 1-14, no augmentation) and 300 whole games, no search (seeds 50900001..,
as V54).  -> eval.json
  uv run --project ../v50-transformer python eval.py NAME ..."""
import json, sys
from pathlib import Path
import numpy as np
import torch
L = Path(__file__).resolve().parent; sys.path.insert(0, str(L.parent / 'v54-symmetry')); sys.path.insert(0, str(L.parent / 'v50-transformer'))
from models import Shared
from train import arrays, on_dev, usable, evaluate, DATA, dev
from play import run
from engine import Game
sys.path.insert(0, str(L.parent / 'v55-shared-dagger')); import shared as S

def sample(d, rows, k, rng):
    idx = np.flatnonzero(rows); m = np.zeros(len(rows), bool); m[rng.choice(idx, min(k, len(idx)), replace=False)] = True; return on_dev(arrays(d, m))

if __name__ == '__main__':
    rng = np.random.default_rng(56); d = dict(np.load(DATA / 'positions.npz')); ok = usable(d)
    va = on_dev(arrays(d, ok & d['val'])); tra = sample(d, ok & ~d['val'], 150000, rng)
    dg = [dict(np.load(DATA / 'dagger' / f'round-{r}.npz')) for r in range(1, 15)]
    trd = {k: np.concatenate([e[k] for e in dg]) for k in dg[0]}; trd = sample(trd, usable(trd), 150000, rng)
    f = L / 'eval.json'; out = json.loads(f.read_text()) if f.exists() else {}
    for name in sys.argv[1:]:
        run_dir = L / 'runs' / name; cfg = json.loads((run_dir / 'config.json').read_text())
        net = Shared(**cfg).to(dev); net.load_state_dict(torch.load(run_dir / 'model.pt', map_location=dev)); net.eval()
        res = {'config': cfg, 'params': sum(p.numel() for p in net.parameters()), 'val': evaluate(net, va), 'trainAI': evaluate(net, tra), 'trainDagger': evaluate(net, trd),
               'log': [json.loads(l)['val'] for l in (run_dir / 'log.jsonl').read_text().splitlines()]}
        def choose(games, net=net):
            t = lambda a, dt=torch.long: torch.as_tensor(np.array(a)).to(dev, dt)
            legal = [[bool(S.move(g.b, k)[1]) for k in range(4)] for g in games]
            with torch.no_grad():
                lo, _ = net(t([g.b for g in games]), t([(g.next['cards'] + [0, 0, 0])[:3] for g in games]), t([g.next['bonus'] for g in games], torch.bool),
                            t([g.counts for g in games]), t([g.cycle for g in games]), t(legal, torch.bool))
            return lo.argmax(1).tolist()
        games = [Game(50900001 + i) for i in range(300)]; run(games, choose); top = [max(g.b) for g in games]
        res['play'] = {'meanScore': float(np.mean([g.score() for g in games])), 'meanMoves': float(np.mean([g.turns for g in games])),
                       **{f'reach{c}': float(np.mean([t >= k for t in top])) for c, k in (('1536', 12), ('3072', 13), ('6144', 14), ('12288', 15))}}
        out[name] = res; print(name, {k: v for k, v in res.items() if k != 'log'}, flush=True)
        f.write_text(json.dumps(out, indent=1) + '\n')
