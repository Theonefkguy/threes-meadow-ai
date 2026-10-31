"""V53 step 4: is a game faster with the transformer + search on the risky moves than with the search on
every move? One game at a time, one core for each part: torch with 1 thread (TORCH_DEVICE=cpu) and one
bin/label process (the website AI's 6-ply search). Same seeds 50900001.. for every mode:
  search    the search on every move (the website AI);
  hybrid30  / hybrid20: the risk score of play.py (8 symmetry frames + signals + risk.pt), the search
            when it is above the validation threshold for the least sure 30% / 20%;
  hybrid30h / hybrid20h: the same with the cheap risk score of risk_h.py (one forward pass);
  model     the transformer alone.
Also a micro benchmark on 500 positions of a model-only game: one forward pass, the 8 frames, the greedy
signals, the risk network (batch 1).  -> bench.json
  TORCH_DEVICE=cpu uv run --project training/v50-transformer python training/v53-confidence/bench.py [GAMES]"""
import copy, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
L = Path(__file__).resolve().parent; sys.path.insert(0, str(L))
from conf import signals, frames, greedy, Risk, inputs, SIG, dev
from common import load_net, to_dev, run, V50
from engine import Game, move
torch.set_num_threads(1)
Q = L.parent / 'web-release' / 'models'
CMD = [str(V50 / 'bin' / 'label'), str(Q / 'ntuple-v31.bin'), ','.join(str(Q / f) for f in ('v31-s2q.res', 'v31-s1q.res', 'v31-s3q.res')), '-', '/dev/stdout']
worker = subprocess.Popen(CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
def search(g):
    worker.stdin.write(' '.join(map(str, [*g.b, len(g.next['cards']), *g.next['cards'], int(g.next['bonus']), *g.counts, *g.cycle])) + '\n'); worker.stdin.flush()
    return int(np.nanargmax([float(x) for x in worker.stdout.readline().split()]))

net = load_net(); R = torch.load(L / 'risk.pt', map_location=dev, weights_only=False); assert R['best'].startswith('network')
rnet = Risk(128 + 4 + len(SIG)).to(dev); rnet.load_state_dict(R['state']); rnet.eval()
RH = torch.load(L / 'risk-h.pt', map_location=dev, weights_only=False); hnet = Risk(128).to(dev); hnet.load_state_dict(RH['state']); hnet.eval()
def x_of(g):
    x = {'board': [g.b], 'cards': [(g.next['cards'] + [0, 0, 0])[:3]], 'bonus': [g.next['bonus']], 'counts': [g.counts], 'cycle': [list(g.cycle)],
         'legal': [[bool(move(g.b, d)[1]) for d in range(4)]]}
    return to_dev({k: np.array(v) for k, v in x.items()})
def risky_h(g, q):
    with torch.no_grad():
        x = x_of(g); lo, res, _, _ = run(net, x); r = hnet(net.norm(res[-1][:, 0])).item()
    return int(lo.argmax()), r > RH['thresholds'][q]
def risky(g, q):
    S, h, P = signals(net, x_of(g))
    with torch.no_grad(): r = rnet(torch.as_tensor(inputs(S, h, P, R['mu'], R['sd']), device=dev)).item()
    return int(P.argmax()), r > R['thresholds'][q]

def game(seed, mode):
    g = Game(seed); n = asked = 0; ts = 0.0; t0 = time.perf_counter()
    while not g.over and g.turns < 50000:
        if mode == 'search': t = time.perf_counter(); d = search(g); ts += time.perf_counter() - t; asked += 1
        elif mode == 'model':
            with torch.no_grad(): d = int(run(net, x_of(g))[0].argmax())
        else:
            d, ask = (risky_h if mode.endswith('h') else risky)(g, {'30': 0.3, '20': 0.2}[mode[6:8]])
            if ask: t = time.perf_counter(); d = search(g); ts += time.perf_counter() - t; asked += 1
        g.step(d); n += 1
    return {'seed': seed, 'moves': n, 'sec': time.perf_counter() - t0, 'searchSec': ts, 'searched': asked, 'top': max(g.b), 'score': g.score()}

def micro(k=500):
    g = Game(50900001); pos = []
    while not g.over and len(pos) < 3000:
        pos.append(copy.deepcopy(g))
        with torch.no_grad(): g.step(int(run(net, x_of(g))[0].argmax()))
    pos = pos[-k:]; xs = [x_of(p) for p in pos]; out = {}
    def timeit(name, f):
        t = time.perf_counter()
        for i, x in enumerate(xs): f(i, x)
        out[name] = (time.perf_counter() - t) / len(xs) * 1000
    with torch.no_grad():
        timeit('forward', lambda i, x: run(net, x))
        timeit('8 frames', lambda i, x: frames(net, x))
        timeit('greedy signals', lambda i, x: (greedy(pos[i].b, 0), greedy(pos[i].b, 1)))
        timeit('all signals', lambda i, x: signals(net, x))
        timeit('risk net', lambda i, x: rnet(torch.zeros(1, 128 + 4 + len(SIG), device=dev)))
    t = time.perf_counter()
    for p in pos: search(p)
    out['search'] = (time.perf_counter() - t) / len(pos) * 1000
    return {k: round(v, 3) for k, v in out.items()}   # ms per position, the last k positions of a model-only game

if __name__ == '__main__':
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    out = {'device': str(dev), 'threads': torch.get_num_threads(), 'micro_ms': micro()}; print(out, flush=True)
    for mode in ('model', 'hybrid20', 'hybrid30', 'hybrid20h', 'hybrid30h', 'search'):
        out[mode] = [game(50900001 + i, mode) for i in range(K)]
        r = out[mode]; m = sum(x['moves'] for x in r)
        print(mode, f"{np.mean([x['sec'] for x in r]):.1f} s/game, {sum(x['sec'] for x in r) / m * 1000:.2f} ms/move, search {sum(x['searchSec'] for x in r) / m * 1000:.2f} ms/move,"
              f" searched {sum(x['searched'] for x in r) / m:.1%}, moves {m / K:.0f}, tops {[x['top'] for x in r]}", flush=True)
        (L / 'bench.json').write_text(json.dumps(out, indent=1) + '\n')
    worker.stdin.close(); worker.wait()
