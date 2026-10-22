"""V50 part 3: PPO fine-tuning of the distilled transformer, playing with no search.
Reward: 1 per move survived (a longer game means bigger cards: every move adds a card to the board),
plus 1/(1-GAMMA) for making 12288 (the game ends there). GAMMA 0.997, GAE lambda 0.95.
Starts: 80% late starts (a random position of a training AI game whose top card is >= 1536, replayed
with the AI's moves, then the RNG is re-seeded so every start has a new future), 20% from the opening.
Policy: the DAgger model; a frozen copy is the reference, and the loss adds BETA * KL(policy || reference).
Value: a separate copy of the transformer with a value head (summary token -> 1), target returns * (1-GAMMA).
The first WARM iterations train only the value. Every EVAL iterations: the late-start test (play.py late)
and a snapshot. Checkpoint every iteration; SIGTERM saves and exits 75.
  uv run python rl.py NAME INIT_MODEL ITERS  ->  runs/NAME/model.pt, runs/NAME/log.jsonl"""
import copy, json, os, random, signal, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from engine import Game, RNG, move
from model import Net
from aistart import ai_moves, ai_seeds
import play
L = Path(__file__).resolve().parent
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
N, T, GAMMA, LAM = 512, 128, 0.997, 0.95
CLIP, BETA, LR, VLR, EPOCHS, MB = 0.2, 0.02, 2e-5, 3e-4, 4, 4096
WARM, EVAL, POOL, SCRATCH = 10, 20, 5, 0.2
if os.environ.get('SMOKE'): N, T, WARM, EVAL, POOL = 64, 16, 1, 2, 1

class ValueNet(nn.Module):
    def __init__(self, init):
        super().__init__(); self.net = Net(); self.net.load_state_dict(init); self.v = nn.Linear(self.net.norm.normalized_shape[0], 1)
    def forward(self, board, cards, bonus, counts, cycle): return self.v(self.net.features(board, cards, bonus, counts, cycle))[:, 0]

def start_pool(seed):
    """POOL late starts per training AI game: random positions with top card >= 1536."""
    rng = random.Random(seed); pool = []
    for s in ai_seeds(False):
        chosen, boards = ai_moves()[s]; k = np.flatnonzero(boards.max(1) >= 12)
        if not len(k): continue
        at = set(rng.sample(list(k), min(POOL, len(k)))); g = Game(s)
        for i, d in enumerate(chosen):
            if i in at: pool.append(copy.deepcopy(g))
            if i >= max(at): break
            g.step(int(d))
    return pool

def encode(games):
    t = lambda a, dt=torch.long: torch.as_tensor(np.array(a)).to(dev, dt)
    legal = [[bool(move(g.b, d)[1]) for d in range(4)] for g in games]
    return (t([g.b for g in games]), t([(g.next['cards'] + [0, 0, 0])[:3] for g in games]), t([g.next['bonus'] for g in games], torch.bool),
            t([g.counts for g in games]), t([g.cycle for g in games])), t(legal, torch.bool)

if __name__ == '__main__':
    name, init, iters = sys.argv[1], sys.argv[2], int(sys.argv[3])
    run = L / 'runs' / name; run.mkdir(parents=True, exist_ok=True); ck = run / 'ckpt.pt'
    sd = torch.load(init, map_location=dev)
    pol, ref = Net().to(dev), Net().to(dev); pol.load_state_dict(sd); ref.load_state_dict(sd); ref.eval()
    vnet = ValueNet(sd).to(dev)
    popt = torch.optim.AdamW(pol.parameters(), lr=LR, weight_decay=0.0); vopt = torch.optim.AdamW(vnet.parameters(), lr=VLR, weight_decay=0.0)
    it0 = 0
    if ck.exists():
        c = torch.load(ck, map_location=dev); pol.load_state_dict(c['pol']); vnet.load_state_dict(c['vnet']); popt.load_state_dict(c['popt']); vopt.load_state_dict(c['vopt']); it0 = c['it']
        print('resumed at iteration', it0, flush=True)
    def save(it): torch.save({'pol': pol.state_dict(), 'vnet': vnet.state_dict(), 'popt': popt.state_dict(), 'vopt': vopt.state_dict(), 'it': it}, ck)
    state = {'it': it0}
    signal.signal(signal.SIGTERM, lambda *a: (save(state['it']), sys.exit(75)))
    t0 = time.time(); pool = start_pool(50); print(f'{len(pool)} late starts ({time.time() - t0:.0f} s)', flush=True)
    rng = random.Random(1000 + it0); nseed = [60000000 + 100000 * it0]
    def reset():
        nseed[0] += 1
        if rng.random() < SCRATCH: g = Game(nseed[0])
        else: g = copy.deepcopy(rng.choice(pool)); g.rng = RNG(nseed[0])
        g.top0 = max(g.b); g.t0 = g.turns; return g
    envs = [reset() for _ in range(N)]; log = open(run / 'log.jsonl', 'a')
    for it in range(it0, iters):
        t0 = time.time(); buf = []; eps = []
        pol.eval()
        with torch.no_grad():
            for _ in range(T):
                obs, legal = encode(envs)
                logits, _ = pol(*obs, legal); rl, _ = ref(*obs, legal); v = vnet(*obs)
                dist = torch.distributions.Categorical(logits=logits); a = dist.sample()
                rew, done = [], []
                for i, (g, d) in enumerate(zip(envs, a.tolist())):
                    g.step(d); top = max(g.b)
                    rew.append(1.0 + (1 / (1 - GAMMA) if top >= 15 else 0.0)); done.append(g.over)
                    if g.over:
                        eps.append((g.top0, top, g.turns - g.t0)); envs[i] = reset()
                buf.append((obs, legal, a, dist.log_prob(a), v, torch.tensor(rew, device=dev), torch.tensor(done, device=dev, dtype=torch.float32)))
            last = vnet(*encode(envs)[0])
        # GAE (values are in units of 1/(1-GAMMA) moves)
        adv = torch.zeros(T, N, device=dev); g_ = torch.zeros(N, device=dev)
        V = torch.stack([b[4] for b in buf]) / (1 - GAMMA); nxt = last / (1 - GAMMA)
        for k in reversed(range(T)):
            r, dn = buf[k][5], buf[k][6]
            delta = r + GAMMA * nxt * (1 - dn) - V[k]; g_ = delta + GAMMA * LAM * (1 - dn) * g_; adv[k] = g_; nxt = V[k]
        ret = (adv + V) * (1 - GAMMA)
        cat = lambda j: torch.cat([b[j] for b in buf])
        O = [torch.cat([b[0][i] for b in buf]) for i in range(5)]; LG, A, LP = cat(1), cat(2), cat(3)
        ADV = adv.flatten(); ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8); RET = ret.flatten()
        pol.train(); stats = {'kl': 0.0, 'clip': 0.0, 'vloss': 0.0, 'n': 0}; M = len(A)
        for _ in range(EPOCHS):
            perm = torch.randperm(M, device=dev)
            for i in range(0, M, MB):
                j = perm[i:i + MB]; o = [x[j] for x in O]
                vl = ((vnet(*o) - RET[j]) ** 2).mean(); vopt.zero_grad(); vl.backward(); nn.utils.clip_grad_norm_(vnet.parameters(), 1.0); vopt.step()
                if it >= WARM:
                    logits, _ = pol(*o, LG[j]); lp = torch.log_softmax(logits, 1)
                    with torch.no_grad(): rlp = torch.log_softmax(ref(*o, LG[j])[0], 1)
                    nlp = lp.gather(1, A[j][:, None])[:, 0]; ratio = (nlp - LP[j]).exp()
                    pg = -torch.min(ratio * ADV[j], ratio.clamp(1 - CLIP, 1 + CLIP) * ADV[j]).mean()
                    kl = (lp.exp() * (lp - rlp)).masked_fill(~LG[j], 0).sum(1).mean()
                    loss = pg + BETA * kl; popt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(pol.parameters(), 1.0); popt.step()
                    stats['kl'] += kl.item(); stats['clip'] += ((ratio - 1).abs() > CLIP).float().mean().item()
                stats['vloss'] += vl.item(); stats['n'] += 1
        state['it'] = it + 1; save(it + 1)
        late = [e for e in eps if e[0] >= 12]
        rec = {'it': it + 1, 'sec': round(time.time() - t0), 'episodes': len(eps), 'meanLen': float(np.mean([e[2] for e in eps])) if eps else None,
               'next': {str(r): float(np.mean([e[1] > r for e in late if e[0] == r])) for r in (12, 13, 14) if any(e[0] == r for e in late)},
               **{k: v / stats['n'] for k, v in stats.items() if k != 'n'}}
        if (it + 1) % EVAL == 0 or it + 1 == iters:
            torch.save(pol.state_dict(), run / f'it-{it + 1}.pt'); pol.eval()
            def choose(games): obs, lg = encode(games); return pol(*obs, lg)[0].argmax(1).tolist()
            with torch.no_grad(): rec['lateTest'] = play.late(f'{name}@{it + 1}', choose, 4)['late']
        log.write(json.dumps(rec) + '\n'); log.flush(); print(json.dumps(rec), flush=True)
    torch.save(pol.state_dict(), run / 'model.pt')
