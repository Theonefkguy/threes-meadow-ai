"""V53 step 3: whole games, the search on the moves the step-2 risk score (risk.pt) marks as least sure.
300 games (seeds 50900001.., as V52 section 7) with the DAgger-14 transformer; on a move whose risk is
above the validation threshold for the least sure 10 / 20 / 30%, the website AI's 6-ply search
(bin/label, 12 processes) chooses instead.  -> play.json
  uv run --project training/v50-transformer python training/v53-confidence/play.py [h]   (SMOKE=1: 10 games)
With 'h': the cheap risk score of risk_h.py (final summary token only) -> play-h.json"""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
L = Path(__file__).resolve().parent; sys.path.insert(0, str(L))
from conf import signals, Risk, inputs, SIG, SIGN, dev
from common import load_net, to_dev, V50
from engine import Game, move
Q = L.parent / 'web-release' / 'models'
CMD = [str(V50 / 'bin' / 'label'), str(Q / 'ntuple-v31.bin'), ','.join(str(Q / f) for f in ('v31-s2q.res', 'v31-s1q.res', 'v31-s3q.res')), '-', '/dev/stdout']
workers = [subprocess.Popen(CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1) for _ in range(12)]
def search(games):
    for k, g in enumerate(games):
        workers[k % 12].stdin.write(' '.join(map(str, [*g.b, len(g.next['cards']), *g.next['cards'], int(g.next['bonus']), *g.counts, *g.cycle])) + '\n')
    for w in workers: w.stdin.flush()
    return [int(np.nanargmax([float(x) for x in workers[k % 12].stdout.readline().split()])) for k in range(len(games))]

net = load_net(); R = torch.load(L / 'risk.pt', map_location=dev, weights_only=False); best = R['best']
if best.startswith('network'):
    rnet = Risk(128 + 4 + len(SIG)).to(dev); rnet.load_state_dict(R['state']); rnet.eval()
    def risk(S, h, P):
        with torch.no_grad(): return rnet(torch.as_tensor(inputs(S, h, P, R['mu'], R['sd']), device=dev)).cpu().numpy()
elif best in SIG: risk = lambda S, h, P: SIGN[best] * S[:, SIG.index(best)]
else: risk = lambda S, h, P: ((S - R['mu']) / R['sd']) @ R['logreg'][0] + R['logreg'][1]
H = sys.argv[1:] == ['h']
if H:
    R = torch.load(L / 'risk-h.pt', map_location=dev, weights_only=False); best = 'network, bad5, summary token only'
    hnet = Risk(128).to(dev); hnet.load_state_dict(R['state']); hnet.eval()
    def risk(S, h, P):
        with torch.no_grad(): return hnet(torch.as_tensor(h, device=dev)).cpu().numpy()

def play(q, n=10 if os.environ.get('SMOKE') else 300):
    games = [Game(50900001 + i) for i in range(n)]; alive = list(games); asked = total = 0; t0 = time.time()
    while alive:
        x = {'board': [g.b for g in alive], 'cards': [(g.next['cards'] + [0, 0, 0])[:3] for g in alive], 'bonus': [g.next['bonus'] for g in alive],
             'counts': [g.counts for g in alive], 'cycle': [list(g.cycle) for g in alive], 'legal': [[bool(move(g.b, d)[1]) for d in range(4)] for g in alive]}
        S, h, P = signals(net, to_dev({k: np.array(v) for k, v in x.items()}))
        ds = P.argmax(1).tolist(); ask = np.flatnonzero(risk(S, h, P) > R['thresholds'][q]).tolist() if q else []
        if ask:
            for k, d in zip(ask, search([alive[k] for k in ask])): ds[k] = d
        asked += len(ask); total += len(alive)
        for g, d in zip(alive, ds): g.step(d)
        alive = [g for g in alive if not g.over and g.turns < 50000]
    top = [max(g.b) for g in games]
    return {'asked': asked / total, 'meanScore': float(np.mean([g.score() for g in games])), 'meanMoves': float(np.mean([g.turns for g in games])),
            **{f'reach{c}': float(np.mean([t >= r for t in top])) for c, r in (('1536', 12), ('3072', 13), ('6144', 14), ('12288', 15))}, 'sec': round(time.time() - t0)}

out = {'risk': best}
for name, q in [('只用模型', 0), ('风险最高的约 10%', 0.1), ('风险最高的约 20%', 0.2), ('风险最高的约 30%', 0.3)]:
    out[name] = play(q); print(name, {k: round(v, 4) for k, v in out[name].items()}, flush=True)
    (L / ('play-smoke.json' if os.environ.get('SMOKE') else 'play-h.json' if H else 'play.json')).write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
for w in workers: w.stdin.close(); w.wait()
