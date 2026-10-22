"""V50: whole games with no search: each move is the highest-scoring legal direction of a policy.
Rules as in the website (engine.py, the C++ game under the bonus-cycle rule); the policy sees only
the public state (board, preview, card counts, cycle). Games run in parallel, one batched model call
per move. Seeds 50900001.. (not used for training). Reports how many games reach each card.
  uv run python play.py POLICY GAMES [OUT.json]
  POLICY: runs/NAME/model.pt (transformer), random, rules (the V49 four-rule list)
Late starts (aistart.py): the 100 validation AI games, from where the AI first had 1536 / 3072;
compared with what the AI itself reached in the same games.
  uv run python play.py POLICY late [OUT.json]"""
import json, sys, time, random
from pathlib import Path
import numpy as np
import torch
from engine import Game, move
from model import Net
L = Path(__file__).resolve().parent
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
FIRST_SEED = 50900001

def transformer(path):
    net = Net().to(dev); net.load_state_dict(torch.load(path, map_location=dev)); net.eval()
    @torch.no_grad()
    def choose(games):
        t = lambda a, dt=torch.long: torch.as_tensor(np.array(a)).to(dev, dt)
        legal = [[bool(move(g.b, d)[1]) for d in range(4)] for g in games]
        logits, _ = net(t([g.b for g in games]), t([(g.next['cards'] + [0, 0, 0])[:3] for g in games]), t([g.next['bonus'] for g in games], torch.bool),
                        t([g.counts for g in games]), t([g.cycle for g in games]), t(legal, torch.bool))
        return logits.argmax(1).tolist()
    return choose

def rules():
    sys.path.insert(0, str(L.parent / 'v49-rules'))
    from features import features, FEATURES
    R = [(FEATURES.index(r['feature']), r['sign']) for r in json.loads((L.parent / 'v49-rules' / 'rules.json').read_text())['rules']['all']]
    rng = random.Random(49)
    def choose(games):
        out = []
        for g in games:
            x = features(g.b, g.next['cards']); k = [not np.isnan(x[d, 0]) for d in range(4)]
            for f, s in R:
                best = max(s * x[d, f] for d in range(4) if k[d]); k = [k[d] and s * x[d, f] == best for d in range(4)]
            out.append(rng.choice([d for d in range(4) if k[d]]))
        return out
    return choose

def uniform():
    rng = random.Random(0)
    return lambda games: [rng.choice([d for d in range(4) if move(g.b, d)[1]]) for g in games]

def run(games, choose):
    alive = [g for g in games if not g.over]
    while alive:
        for g, d in zip(alive, choose(alive)): g.step(d)
        alive = [g for g in alive if not g.over and g.turns < 50000]

def late(pol, choose):
    from aistart import late_games, ai_seeds
    res = {'policy': pol, 'late': {}}
    for r, c in ((12, '1536'), (13, '3072')):
        lg = late_games(ai_seeds(True), r); run([g for g, _ in lg], choose)
        res['late'][c] = {'games': len(lg), **{f'{who} reach {x}': sum(t >= k for t in tops) / len(lg) for who, tops in (('student', [max(g.b) for g, _ in lg]), ('AI', [t for _, t in lg]))
                                               for x, k in (('3072', 13), ('6144', 14), ('12288', 15)) if k > r and not (who == 'AI' and k == 15)}}
    return res

if __name__ == '__main__':
    pol = sys.argv[1]
    choose = uniform() if pol == 'random' else rules() if pol == 'rules' else transformer(pol)
    if sys.argv[2] == 'late':
        res = late(pol, choose); print(json.dumps(res))
        if len(sys.argv) > 3: Path(sys.argv[3]).write_text(json.dumps(res) + '\n')
        sys.exit(0)
    n = int(sys.argv[2])
    games = [Game(FIRST_SEED + i) for i in range(n)]; t0 = time.time()
    run(games, choose)
    top = [max(g.b) for g in games]
    reach = {c: sum(t >= r for t in top) / n for c, r in (('384', 10), ('768', 11), ('1536', 12), ('3072', 13), ('6144', 14), ('12288', 15))}
    res = {'policy': pol, 'games': n, 'reach': reach, 'meanScore': sum(g.score() for g in games) / n, 'meanMoves': sum(g.turns for g in games) / n,
           'sec': round(time.time() - t0), 'top': top}
    print(json.dumps({k: v for k, v in res.items() if k != 'top'}))
    if len(sys.argv) > 3: Path(sys.argv[3]).write_text(json.dumps(res) + '\n')
