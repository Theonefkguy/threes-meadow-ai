"""V62 analysis (setup in RL-V62-TIME-SHARE.md): where the 6-ply / 0.005 search time goes over a whole website-AI
game, and how much "5-ply / 0.01 when there are >= k empty cells" would save; plus how dangerous positions with
k empty cells are (death within 30 moves, V61 step-1 records of the 1000 games).  -> results.json"""
import json
from pathlib import Path
import numpy as np
L = Path(__file__).resolve().parent
meta = json.loads((L / 'positions.json').read_text())
T = np.loadtxt(L / 'timing.txt'); T = T[np.argsort(T[:, 0])]
top = np.array([m['top'] for m in meta]); empty = np.array([m['empty'] for m in meta])
ms6, ms5, same = T[:, 1], T[:, 2], T[:, 3] == T[:, 4]
stage = np.where(top >= 14, '6144+', np.where(top == 13, '3072', '<=1536'))

def table(sel):
    tot = ms6[sel].sum(); out = {}
    for e in range(9):
        s = sel & (empty == e)
        if s.sum(): out[e] = {'moveShare': float(s.sum() / sel.sum()), 'ms6': float(ms6[s].mean()), 'ms5': float(ms5[s].mean()),
                              'timeShare': float(ms6[s].sum() / tot), 'sameDir': float(same[s].mean())}
    save = {k: {'movesSwitched': float((sel & (empty >= k)).sum() / sel.sum()),
                'timeSaved': float((ms6 - ms5)[sel & (empty >= k)].sum() / tot)} for k in range(1, 9)}
    return {'moves': int(sel.sum()), 'ms6': float(ms6[sel].mean()), 'ms5': float(ms5[sel].mean()), 'byEmpty': out, 'switchAtLeast': save}

out = {'all': table(np.ones(len(T), bool))}
for g in ('<=1536', '3072', '6144+'):
    out[g] = table(stage == g); out[g]['shareOfGameMoves'] = float((stage == g).mean()); out[g]['shareOfGameTime'] = float(ms6[stage == g].sum() / ms6.sum())

# danger: death within 30 moves by empty cells before the move (steps 'empty' is after the move, before the new card)
S = np.load('/Volumes/Extreme SSD/Datasets/threes-v61/steps.npz'); seed, move, e_after, over, tp = S['seed'], S['move'], S['empty'], S['over'], S['top']
o = np.lexsort((move, seed)); seed, move, e_after, over, tp = seed[o], move[o], e_after[o], over[o], tp[o]
first = np.r_[True, seed[1:] != seed[:-1]]; e_before = np.r_[0, e_after[:-1] - 1]; keep = ~first
endIdx = np.r_[np.nonzero(seed[1:] != seed[:-1])[0], len(seed) - 1]; gi = np.cumsum(first) - 1
lastMove, lost = move[endIdx][gi], (tp[endIdx] < 15)[gi]
d30 = lost & (lastMove - move < 30); late = tp >= 14
out['danger30'] = {g: {int(e): {'share': float((sel & (e_before == e)).sum() / sel.sum()), 'death30': float(d30[sel & (e_before == e)].mean())}
                       for e in range(9) if (sel & (e_before == e)).sum() >= 100}
                   for g, sel in (('all', keep), ('6144+', keep & late))}
(L / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
for g in ('all', '6144+'):
    r = out[g]; print(f"== {g}: {r['moves']} moves, 6-ply {r['ms6']:.1f} ms, 5-ply {r['ms5']:.1f} ms")
    for e, x in r['byEmpty'].items():
        print(f"  empty {e}: moves {x['moveShare']:.3f}  ms6 {x['ms6']:6.1f}  ms5 {x['ms5']:5.1f}  time {x['timeShare']:.3f}  sameDir {x['sameDir']:.3f}  death30 {out['danger30'][g].get(e, {}).get('death30', float('nan')):.4f}")
    for k, x in r['switchAtLeast'].items(): print(f"  >= {k}: switch {x['movesSwitched']:.3f} save {x['timeSaved']:.3f}")
for g in ('<=1536', '3072', '6144+'): print(g, f"moves {out[g]['shareOfGameMoves']:.3f} time {out[g]['shareOfGameTime']:.3f} ms6 {out[g]['ms6']:.1f}")
