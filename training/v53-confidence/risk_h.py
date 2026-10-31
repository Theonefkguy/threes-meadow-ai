"""V53 step 4b: the cheap risk score, the Risk network on the final summary token alone (no symmetry
frames, no signals; cross-validated AUC 0.92 in signals.json), fitted on all validation positions.
  uv run --project training/v50-transformer python training/v53-confidence/risk_h.py  -> risk-h.pt"""
import sys
from pathlib import Path
import numpy as np
import torch
L = Path(__file__).resolve().parent; sys.path.insert(0, str(L))
from signals import fit_risk
from common import load_net, sample, to_dev, run, gaps

net = load_net(); a = sample(10 ** 7, seed=53); n = len(a['board']); H, P = [], []
for i in range(0, n, 4096):
    with torch.no_grad():
        lo, res, _, _ = run(net, to_dev(a, slice(i, i + 4096))); H.append(net.norm(res[-1][:, 0]).cpu().numpy()); P.append(lo.argmax(1).cpu().numpy())
H, P = np.concatenate(H), np.concatenate(P); bad = gaps(a['values'])[np.arange(n), P] >= 0.05
rnet, fn = fit_risk(H.astype(np.float32), bad, 'bad5'); s = fn(H.astype(np.float32))
th = {q: float(np.quantile(s, 1 - q)) for q in (0.1, 0.2, 0.3)}
torch.save({'state': rnet.state_dict(), 'thresholds': th}, L / 'risk-h.pt'); print(n, 'positions, thresholds', th)
