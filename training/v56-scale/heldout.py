"""V56: s128 and s256 on DAgger rounds neither model was trained on, collected by V55's shared students
(round 15: by the V54 shared, round 22: by V55 round 21), 300k positions at most per round, no augmentation.
The validation set holds the website AI's own games; these rounds hold positions the students reach.
  uv run --project ../v50-transformer python heldout.py NAME... -> heldout.json"""
import json, sys
from pathlib import Path
import numpy as np
import torch
L = Path(__file__).resolve().parent; sys.path.insert(0, str(L.parent / 'v54-symmetry')); sys.path.insert(0, str(L.parent / 'v50-transformer'))
from models import Shared
from train import arrays, on_dev, usable, evaluate, DATA, dev

out = {}
for r in (15, 22):
    e = dict(np.load(DATA / 'dagger' / f'round-{r}.npz')); ok = usable(e); idx = np.flatnonzero(ok)
    m = np.zeros(len(ok), bool); m[np.sort(np.random.default_rng(56).choice(idx, min(len(idx), 300000), replace=False))] = True
    x = on_dev(arrays(e, m))
    for name in sys.argv[1:]:
        c = json.loads((L / 'runs' / name / 'config.json').read_text())
        net = Shared(d=c['d'], layers=c['layers'], heads=c['heads'], ff=c['ff']).to(dev)
        net.load_state_dict(torch.load(L / 'runs' / name / 'model.pt', map_location=dev)); net.eval()
        out.setdefault(name, {})[f'round{r}'] = evaluate(net, x); print(name, r, {k: round(v, 4) for k, v in out[name][f'round{r}'].items()}, flush=True)
(L / 'heldout.json').write_text(json.dumps(out, indent=1) + '\n')
