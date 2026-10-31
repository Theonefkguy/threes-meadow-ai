"""V61 step 1 analysis (setup in RL-V61-CARD-LUCK.md). A: the last move of the 885 lost games (last.json).
B: the last K cards before death against windows of the same stage that are followed by >= 100 more moves;
death - control with a 95% bootstrap interval over games (2000 resamples).  -> results.json"""
import json
from pathlib import Path
import numpy as np
L = Path(__file__).resolve().parent
S = np.load('/Volumes/Extreme SSD/Datasets/threes-v61/steps.npz'); last = json.loads((L / 'last.json').read_text())
STAGES = (('≤1536', 0, 12), ('3072', 13, 13), ('6144', 14, 14))
FEATS = ['n1', 'n2', 'n3', 'nBonus', 'd12', 'absD12', 'maxRun', 'wrongSmall', 'emptyStart', 'empty', 'boardOnes', 'boardTwos']

def stage(top): return next(k for k, lo, hi in STAGES if lo <= top <= hi)

# ---- A: last move
A = {'lost': len(last)}
cat = np.array(['otherCell' if r['otherCell'] else 'otherCard' if r['otherCard'] else 'anything' if r['anything'] else 'none' for r in last])
A['onlyCellOrCard'] = {k: float((cat == k).mean()) for k in ('otherCell', 'otherCard', 'anything', 'none')}
A['otherCell'] = float(np.mean([r['otherCell'] for r in last])); A['otherCard'] = float(np.mean([r['otherCard'] for r in last]))
A['singleEntry'] = float(np.mean([r['entries'] == 1 for r in last]))
pd = np.array([r['pDeath'] for r in last]); mpd = np.array([r['minPDeath'] for r in last])
A['pDeathChosen'] = {'mean': float(pd.mean()), 'median': float(np.median(pd)), 'below0.1': float((pd < 0.1).mean()), 'below0.25': float((pd < 0.25).mean()), 'atLeast0.5': float((pd >= 0.5).mean())}
A['hadSaferDir'] = float((mpd < pd - 1e-9).mean()); A['allDirsRisky'] = float((mpd > 0).mean())
A['killingCard'] = {str(c): int(sum(r['card'] == c and not r['bonus'] for r in last)) for c in (1, 2, 3)}
A['killingCard']['bonus'] = int(sum(r['bonus'] for r in last))
A['byStage'] = {k: int(sum(stage(r['top']) == k for r in last)) for k, _, _ in STAGES}

# ---- B: windows
seed = S['seed']; order = np.lexsort((S['move'], seed)); assert (order == np.arange(len(seed))).all()
starts = np.r_[0, np.flatnonzero(np.diff(seed)) + 1]; ends = np.r_[starts[1:], len(seed)]
lostSeeds = {r['seed'] for r in last}; G = len(starts)
card, top, empty, ones, twos = S['card'], S['top'], S['empty'], S['ones'], S['twos']

def feats(a, b):                                          # window = steps a .. b-1 of one game (its arrivals)
    c = card[a:b]; o = ones[a:b]; t = twos[a:b]
    run = best = 0; side = 0
    for x in c:
        if x in (1, 2):
            if x == side: run += 1
            else: side, run = x, 1
            best = max(best, run)
    small = (c == 1) | (c == 2); wrong = ((c == 1) & (o > t)) | ((c == 2) & (t > o))
    n1, n2 = int((c == 1).sum()), int((c == 2).sum())
    return [n1, n2, int((c == 3).sum()), int(((c >= 4) & (c < 15)).sum()), n1 - n2, abs(n1 - n2), best,
            wrong.sum() / max(small.sum(), 1), empty[a], empty[b - 1], ones[b - 1], twos[b - 1]]

rng = np.random.default_rng(61); B = {}
for K in (10, 30, 100):
    death = {k: [] for k, _, _ in STAGES}; ctrl = {k: [] for k, _, _ in STAGES}
    for g in range(G):
        a0, e = starts[g], ends[g]; n = e - a0
        if seed[a0] in lostSeeds and n >= K:
            death[stage(top[e - 1])].append((g, feats(e - K, e)))
        for t in range(a0 + K, e - 100, 5):                # windows ending at step t-1, >= 100 more moves after
            ctrl[stage(top[t - 1])].append((g, feats(t - K, t)))
    if K == 10:
        deathK10, ctrlK10 = death, ctrl
        winD = [(ends[g] - K, ends[g]) for k in ('3072', '6144') for g, _ in death[k]]
        winC = [(t - 1, t) for g in range(G) for t in range(starts[g] + K, ends[g] - 100, 5) if top[t - 1] >= 13]   # one card per control window
    B[K] = {}
    for k, _, _ in STAGES:
        if len(death[k]) < 20: continue
        dg = np.array([x[0] for x in death[k]]); dX = np.array([x[1] for x in death[k]], float)
        cg = np.array([x[0] for x in ctrl[k]]); cX = np.array([x[1] for x in ctrl[k]], float)
        dS = np.zeros((G, len(FEATS))); dN = np.zeros(G); np.add.at(dS, dg, dX); np.add.at(dN, dg, 1)
        cS = np.zeros((G, len(FEATS))); cN = np.zeros(G); np.add.at(cS, cg, cX); np.add.at(cN, cg, 1)
        W = rng.multinomial(G, np.ones(G) / G, size=2000).astype(float)
        diff = (W @ dS) / (W @ dN)[:, None] - (W @ cS) / (W @ cN)[:, None]
        B[K][k] = {'deathWindows': len(dX), 'controlWindows': len(cX),
                   **{f: {'death': float(dX[:, j].mean()), 'control': float(cX[:, j].mean()), 'diff': float(dX[:, j].mean() - cX[:, j].mean()),
                          'ci95': [float(np.percentile(diff[:, j], 2.5)), float(np.percentile(diff[:, j], 97.5))]} for j, f in enumerate(FEATS)}}
# likelihood ratios, last 10 cards, stages 3072 and 6144 pooled: P(value | death window) / P(value | control window)
LR = {}
for f in ('nBonus', 'absD12', 'maxRun'):
    j = FEATS.index(f); d = np.r_[[x[1][j] for k in ('3072', '6144') for x in deathK10[k]]]; c = np.r_[[x[1][j] for k in ('3072', '6144') for x in ctrlK10[k]]]
    LR[f] = {int(v): {'death': float((d == v).mean()), 'control': float((c == v).mean()), 'deathN': int((d == v).sum()),
                      'lr': float((d == v).mean() / (c == v).mean())} for v in np.unique(np.r_[d, c]) if (c == v).mean() > 0.003 or (d == v).mean() > 0.01}
# bonus cards in the last 10 cards (stages >= 3072): how far below the top card (in ranks), death against control
def bgap(windows): return [int(top[a + i] - card[a + i]) for a, b in windows for i in range(b - a) if 4 <= card[a + i] < 15]
gd, gc = bgap(winD), bgap(winC)
LR['bonusRanksBelowTop'] = {int(v): {'death': gd.count(v) / len(gd), 'control': gc.count(v) / len(gc)} for v in sorted(set(gd + gc))}
out = {'A': A, 'B': B, 'lastTen': LR}
(L / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
print(json.dumps(A, ensure_ascii=False, indent=1)); print(json.dumps(LR, ensure_ascii=False))
for K in B:
    for k in B[K]:
        print(f'K={K} {k} death {B[K][k]["deathWindows"]} ctrl {B[K][k]["controlWindows"]}')
        for f in FEATS: v = B[K][k][f]; print(f'   {f:10s} death {v["death"]:7.3f} ctrl {v["control"]:7.3f} diff {v["diff"]:+7.3f} [{v["ci95"][0]:+.3f}, {v["ci95"][1]:+.3f}]')
