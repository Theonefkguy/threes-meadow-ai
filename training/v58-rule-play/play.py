"""V58: whole games played by simple rules, no search. 500 games (seeds 50900001.., as V55 eval / shallow.py)
for each player: random; one rule (most monotone lines); V49's four-rule list; V49's point system (all 20
features, score.json 'all'); the V57 "motto" list (no instant death, fewest big cards moved, most monotone
lines, fewest trapped small cards, most merges). Ordered lists keep the best directions rule by rule and
break remaining ties at random (per-game RNG). Features: V49 features.py on the board after the move.
  uv run --project training/v50-transformer python training/v58-rule-play/play.py [GAMES]  -> play.json"""
import json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
L = Path(__file__).resolve().parent; T = L.parent
sys.path.insert(0, str(T / 'v49-rules')); sys.path.insert(0, str(T / 'v50-transformer'))
import features as F49
from engine import Game
FI = {f: i for i, f in enumerate(F49.FEATURES)}
W = json.loads((T / 'v49-rules' / 'score.json').read_text())['all']['weights']; WV = np.array([W.get(f, 0.0) for f in F49.FEATURES])   # fitted on the first 20 features
LISTS = {'一条规则': [('mono', 1)], '四条规则（V49）': [('mono', 1), ('trapped', -1), ('fit', 1), ('setup', 1)],
         '口诀（V57）': [('safe', 1), ('movedBig', -1), ('mono', 1), ('trapped', -1), ('merges', 1)]}
PLAYERS = ['随机', '一条规则', '四条规则（V49）', '打分制（V49）', '口诀（V57）']

def choose(name, g, rng):
    X = F49.features(g.b, [c for c in g.next['cards'] if c]); ok = ~np.isnan(X[:, 0])
    if name == '随机': return int(rng.choice(np.flatnonzero(ok)))
    if name == '打分制（V49）':
        s = np.where(ok, np.nan_to_num(X) @ WV, -np.inf); return int(rng.choice(np.flatnonzero(np.isclose(s, s.max(), rtol=0, atol=1e-9))))
    k = ok.copy()
    for f, sgn in LISTS[name]:
        v = np.where(k, sgn * X[:, FI[f]], -np.inf); k &= v == v.max()
    return int(rng.choice(np.flatnonzero(k)))

def game(args):
    name, seed = args; g = Game(seed); rng = np.random.default_rng(seed)
    while not g.over and g.turns < 50000: g.step(choose(name, g, rng))
    return g.score(), g.turns, max(g.b)

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500; out = {}
    with Pool(12) as p:
        for name in PLAYERS:
            t0 = time.time(); r = p.map(game, [(name, 50900001 + i) for i in range(n)], chunksize=4)
            sc, mv, top = map(np.array, zip(*r))
            out[name] = {'games': n, 'meanScore': float(sc.mean()), 'medianScore': float(np.median(sc)), 'meanMoves': float(mv.mean()),
                         **{f'reach{3 * 2 ** (k - 3)}': float((top >= k).mean()) for k in range(7, 16)}, 'sec': round(time.time() - t0)}
            print(name, {k: round(v, 3) for k, v in out[name].items()}, flush=True)
            (L / ('play.json' if n == 500 else 'play-smoke.json')).write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
