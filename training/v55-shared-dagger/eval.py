"""V55 evaluation of one round's model: 500 whole games (seeds 50900001.., no search; the first 300 are
V54's games), the late-start test (the 100 validation AI games from where the AI first had 1536 / 3072,
each start played 5 times) and the validation numbers of the fine-tune's last epoch.
Round 14 = the starting point, V54's shared model trained from scratch.
Then the stopping rule (fixed before the first round): a round improves if validation 'near' is at least
0.3 points above the best so far, or the 500-game 3072 rate at least 3 points above the best so far;
stop after 2 rounds in a row without improvement, or after round 24.
Added after round 17 (its 6144 rate was still rising): the late-game rule. Late 6144 = the share of games that
reach 6144, pooled over the 500 whole games and the two late-start tests (about 1480 games); a round improves
on it if it is at least 2 points above the best so far. Stop only when both rules have had 2 rounds in a row
without improvement (or after round 24).  -> results/rROUND.json, results/summary.json
  uv run --project ../v50-transformer python eval.py ROUND      (prints go / stop as the last line)"""
import json, sys, time
from pathlib import Path
import numpy as np
from shared import transformer, L
from play import run, late, FIRST_SEED
from engine import Game
LAST, NEAR, R3072, LATE6 = 24, 0.003, 0.03, 0.02

def late6144(x):
    """share of games reaching 6144: 500 whole games + late starts from 1536 and from 3072"""
    parts = [(x['reach6144'], len(x['top']))] + [(x['late'][c]['student reach 6144'], x['late'][c]['games']) for c in ('1536', '3072')]
    return sum(p * n for p, n in parts) / sum(n for _, n in parts)

def judge(res):
    best = best6 = None; flat = flat6 = 0; rows = []
    for r in sorted(res):
        x = res[r]; l6 = late6144(x)
        better = best is None or x['near'] >= best[0] + NEAR or x['reach3072'] >= best[1] + R3072
        better6 = best6 is None or l6 >= best6 + LATE6
        flat = 0 if better else flat + 1; best = (max(x['near'], best[0]), max(x['reach3072'], best[1])) if best else (x['near'], x['reach3072'])
        flat6 = 0 if better6 else flat6 + 1; best6 = l6 if best6 is None else max(best6, l6)
        rows.append({'round': r, 'near': x['near'], 'bad5': x['bad5'], 'reach3072': x['reach3072'], 'reach6144': x['reach6144'], 'late6144': l6,
                     'meanScore': x['meanScore'], 'improves': better, 'improvesLate': better6})
    return rows, ('stop' if (flat >= 2 and flat6 >= 2) or max(res) >= LAST else 'go')

if __name__ == '__main__':
    R = int(sys.argv[1]); out = L / 'results' / f'r{R}.json'
    if not out.exists():
        run_dir = L.parent / 'v54-symmetry' / 'runs' / 'shared' if R == 14 else L / 'runs' / f'r{R}'
        val = json.loads((run_dir / 'log.jsonl').read_text().splitlines()[-1])
        choose = transformer(run_dir / 'model.pt'); t0 = time.time()
        games = [Game(FIRST_SEED + i) for i in range(500)]; run(games, choose)
        top = [max(g.b) for g in games]; lt = late(str(run_dir / 'model.pt'), choose, 4)['late']
        res = {'round': R, 'near': val['val']['near'], 'bad5': val['val']['bad5'], 'val': val['val'], 'v47': val['v47'], 'v48': val['v48'],
               'meanScore': float(np.mean([g.score() for g in games])), 'meanMoves': float(np.mean([g.turns for g in games])),
               **{f'reach{c}': float(np.mean([t >= k for t in top])) for c, k in (('1536', 12), ('3072', 13), ('6144', 14), ('12288', 15))},
               'reach3072_first300': float(np.mean([t >= 13 for t in top[:300]])), 'late': lt,
               'top': top, 'score': [g.score() for g in games], 'sec': round(time.time() - t0)}
        out.write_text(json.dumps(res) + '\n')
    res = {int(p.stem[1:]): json.loads(p.read_text()) for p in (L / 'results').glob('r*.json')}
    rows, verdict = judge(res)
    (L / 'results' / 'summary.json').write_text(json.dumps({'rounds': rows, 'next': verdict}, indent=1) + '\n')
    for x in rows: print(x, flush=True)
    print(verdict)
