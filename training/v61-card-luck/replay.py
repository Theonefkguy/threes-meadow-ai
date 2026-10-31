"""V61 step 1: replay the 1000 website-AI games of V46 / V50 with the V50 Python engine and record, for every
move, the card that actually entered (with the bonus value), where it landed, and the board before the move
(empty cells, top card) and the 1s and 2s on the board after the move, before the new card enters.
For the last move of a lost game, also whether another landing cell or another ordinary card would have survived.
  -> /Volumes/Extreme SSD/Datasets/threes-v61/steps.npz, last.json"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
L = Path(__file__).resolve().parent; T = L.parent
sys.path.insert(0, str(T / 'v50-transformer'))
from engine import Game, move, legal
DIRS = [T / 'v46-margins' / 'data', Path('/Volumes/Extreme SSD/Datasets/threes-v50/data')]
OUT = Path('/Volumes/Extreme SSD/Datasets/threes-v61')

def one(f):
    rows, last = [], []
    cur = None; prev = None
    for line in f.open():
        m = json.loads(line)
        if m['move'] == 0: cur = Game(m['seed'])
        assert cur.b == m['board'] and cur.next['cards'] == m['next']['cards'], (m['seed'], m['move'])
        b = cur.b; d = m['chosen']; a, entries = move(b, d); bonus = cur.next['bonus']
        ones, twos = a.count(1), a.count(2)                                  # after the move, before the card
        cur.step(d)
        if max(a) >= 15: cell, card = -1, 15                                  # the 12288 merge: no card enters
        else:
            cell = next(i for i in range(16) if cur.b[i] != a[i]); card = cur.b[cell]
        rows.append((m['seed'], m['move'], max(b), b.count(0), ones, twos, card, int(bonus), len(entries), cell,
                     int(cur.over), m['dirs'][d]['pDeath']))
        if cur.over and max(cur.b) < 15:                                     # lost: what if the card or cell differed?
            def lives(cl, cd): x = list(a); x[cl] = cd; return legal(x)
            otherCell = any(lives(c, card) for c in entries if c != cell)
            otherCard = any(lives(cell, c) for c in (1, 2, 3) if c != card)
            anything = any(lives(c, cd) for c in entries for cd in (1, 2, 3, card))
            legalDirs = [k for k in range(4) if m['dirs'][k]['legal']]
            last.append({'seed': m['seed'], 'move': m['move'], 'top': max(b), 'card': card, 'bonus': bool(bonus), 'entries': len(entries),
                         'otherCell': otherCell, 'otherCard': otherCard, 'anything': anything, 'pDeath': m['dirs'][d]['pDeath'],
                         'minPDeath': min(m['dirs'][k]['pDeath'] for k in legalDirs), 'legalDirs': len(legalDirs), 'empty': b.count(0)})
    return rows, last

if __name__ == '__main__':
    files = sorted(f for D in DIRS for f in D.glob('m-*.jsonl'))
    with Pool(12) as p: res = p.map(one, files, chunksize=1)
    rows = [r for x in res for r in x[0]]; last = [r for x in res for r in x[1]]
    cols = ['seed', 'move', 'top', 'empty', 'ones', 'twos', 'card', 'bonus', 'entries', 'cell', 'over', 'pDeath']
    A = np.array(rows, dtype=np.float64)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / 'steps.npz', **{c: A[:, k].astype(np.float32 if c == 'pDeath' else np.int64) for k, c in enumerate(cols)})
    (L / 'last.json').write_text(json.dumps(last) + '\n')
    print(len(files), 'files', len(rows), 'moves', len(last), 'lost games')
