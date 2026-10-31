"""V55: is the shared model short of capacity? Its validation numbers on data it was trained on (V46 AI
training games, DAgger rounds), on data it was not (V46 validation games, the newest DAgger round, labelled
after this model was trained), 300k positions at most per set, no symmetry augmentation.
If training ~ held-out, the model underfits (capacity or training time is the limit, not data).
  uv run --project ../v50-transformer python fitcheck.py MODEL NEW_ROUND [TRAINED_ROUNDS...]  -> printed"""
import sys
from pathlib import Path
import numpy as np
import torch
L = Path(__file__).resolve().parent; sys.path.insert(0, str(L.parent / 'v54-symmetry')); sys.path.insert(0, str(L.parent / 'v50-transformer'))
from models import Shared
from train import arrays, on_dev, usable, evaluate, DATA, dev
net = Shared().to(dev); net.load_state_dict(torch.load(sys.argv[1], map_location=dev)); net.eval()
rng = np.random.default_rng(55)
def show(name, d, rows):
    idx = np.flatnonzero(rows); idx = np.sort(rng.choice(idx, min(len(idx), 300000), replace=False))
    m = np.zeros(len(rows), bool); m[idx] = True
    print(f'{name:34s}', {k: round(v, 4) for k, v in evaluate(net, on_dev(arrays(d, m))).items()}, flush=True)
d = dict(np.load(DATA / 'positions.npz')); ok = usable(d)
show('AI games, training (trained)', d, ok & ~d['val']); show('AI games, validation (held out)', d, ok & d['val'])
for r in map(int, sys.argv[3:]):
    e = dict(np.load(DATA / 'dagger' / f'round-{r}.npz')); show(f'DAgger round {r} (trained)', e, usable(e))
e = dict(np.load(DATA / 'dagger' / f'round-{sys.argv[2]}.npz')); show(f'DAgger round {sys.argv[2]} (held out)', e, usable(e))
