"""V55: the V54 shared model as a V50 play.py policy, and V50 dagger.py run with it as the student.
  uv run --project ../v50-transformer python shared.py dagger ROUND MODEL   (400 games + 800 late starts, 12 label processes)"""
import runpy, sys
from pathlib import Path
import numpy as np
import torch
L = Path(__file__).resolve().parent; V50 = L.parent / 'v50-transformer'
sys.path.insert(0, str(L.parent / 'v54-symmetry')); sys.path.insert(0, str(V50))
from models import Shared
import play
from engine import move
dev = play.dev

def transformer(path):
    net = Shared().to(dev); net.load_state_dict(torch.load(path, map_location=dev)); net.eval()
    @torch.no_grad()
    def choose(games):
        t = lambda a, dt=torch.long: torch.as_tensor(np.array(a)).to(dev, dt)
        legal = [[bool(move(g.b, d)[1]) for d in range(4)] for g in games]
        logits, _ = net(t([g.b for g in games]), t([(g.next['cards'] + [0, 0, 0])[:3] for g in games]), t([g.next['bonus'] for g in games], torch.bool),
                        t([g.counts for g in games]), t([g.cycle for g in games]), t(legal, torch.bool))
        return logits.argmax(1).tolist()
    return choose

if __name__ == '__main__' and sys.argv[1] == 'dagger':
    play.transformer = transformer                              # dagger.py takes the student from play.transformer
    sys.argv = ['dagger.py', sys.argv[2], sys.argv[3], '400', '12', '800']
    runpy.run_path(str(V50 / 'dagger.py'), run_name='__main__')
