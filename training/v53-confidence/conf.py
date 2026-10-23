"""V53 shared code: confidence signals of the V50 DAgger-14 transformer for a batch of positions.
signals(net, x) -> (S [B, len(SIG)] float32, h [B,128] final normed summary token, P [B,4] probabilities).
SIG (raw values; SIGN[k] = +1 when a larger value should mean "less sure"):
  pmax, margin (top two probabilities), entropy;
  symDisagree (share of the 8 rotations / mirrors whose choice differs from the model's), symMeanP and
  symStdP (probability of the model's choice across the 8 frames);
  phase (top rank), empties;
  dMerges, dMergeMax, dSetup, dSetupBig: chosen minus second-choice direction (V49 move: merges, largest
  merged card, largest merge available next move, number of next directions merging a card of 24+);
  0 when only one direction is legal.
Run from the repo root:  uv run --project training/v50-transformer python training/v53-confidence/X.py"""
import sys
from pathlib import Path
import numpy as np
import torch
L = Path(__file__).resolve().parent
sys.path.insert(0, str(L.parent / 'v51-interpret')); sys.path.insert(0, str(L.parent / 'v49-rules'))
from common import run, SYM, dev
import features as F49

SIG = ['pmax', 'margin', 'entropy', 'symDisagree', 'symMeanP', 'symStdP', 'phase', 'empties', 'dMerges', 'dMergeMax', 'dSetup', 'dSetupBig']
SIGN = dict(pmax=-1, margin=-1, entropy=1, symDisagree=1, symMeanP=-1, symStdP=1, phase=1, empties=-1, dMerges=1, dMergeMax=1, dSetup=1, dSetupBig=1)
CELL = torch.tensor([s[0] for s in SYM], device=dev); DIRS = torch.tensor([s[1] for s in SYM], device=dev)

@torch.no_grad()
def frames(net, x):
    """probabilities in the original frame for each of the 8 symmetries: [8, B, 4]"""
    out = []
    for k in range(8):
        y = dict(x); y['board'] = x['board'][:, CELL[k]]; y['legal'] = torch.zeros_like(x['legal']).scatter(1, DIRS[k].expand(len(x['legal']), -1), x['legal'])
        out.append(torch.softmax(run(net, y)[0], 1)[:, DIRS[k]])
    return torch.stack(out)

def greedy(b, d):
    a, _, entries, merges, top = F49.move(b, d)
    return merges, top, max(F49.move(a, d2)[4] for d2 in range(4)), sum(F49.move(a, d2)[4] >= 6 for d2 in range(4))

@torch.no_grad()
def signals(net, x):
    logits, res, _, _ = run(net, x); h = net.norm(res[-1][:, 0]); P = torch.softmax(logits, 1)
    Fr = frames(net, x); B = len(P); pick = P.argmax(1); ar = torch.arange(B, device=dev)
    ps = P.sort(1, descending=True).values
    ent = -(P * P.clamp_min(1e-12).log()).sum(1)
    fp = Fr[:, ar, pick]                                                   # [8,B]
    S = torch.stack([ps[:, 0], ps[:, 0] - ps[:, 1], ent, (Fr.argmax(2) != pick).float().mean(0), fp.mean(0), fp.std(0),
                     x['board'].max(1).values.float(), (x['board'] == 0).sum(1).float()], 1).cpu().numpy()
    second = P.masked_fill(F_onehot(pick), -1).argmax(1).cpu().numpy(); pk = pick.cpu().numpy()
    nleg = x['legal'].sum(1).cpu().numpy(); boards = x['board'].cpu().numpy().tolist()
    G = np.zeros((B, 4))
    for i in range(B):
        if nleg[i] > 1: G[i] = np.subtract(greedy(boards[i], int(pk[i])), greedy(boards[i], int(second[i])))
    return np.concatenate([S, G], 1).astype(np.float32), h.cpu().numpy(), P.cpu().numpy()

def F_onehot(pick): return torch.nn.functional.one_hot(pick, 4).bool()

class Risk(torch.nn.Module):
    """the step-2 network: [h, sorted P, standardized S] -> one risk score."""
    def __init__(self, n_in, hidden=64):
        super().__init__(); self.f = torch.nn.Sequential(torch.nn.Linear(n_in, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 1))
    def forward(self, z): return self.f(z)[:, 0]

def inputs(S, h, P, mu, sd):
    return np.concatenate([h, -np.sort(-P, 1), (S - mu) / sd], 1).astype(np.float32)
