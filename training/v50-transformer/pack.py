"""V50: pack website-AI move records (margins JSON lines) into arrays for training.
Sources: V46 (training/v46-margins/data, 200 games) and V50 (external disk, 800 games).
Split by game: validation = V46 odd seeds (the V49 test games), training = everything else.
Also packs the human positions of V47 / V48 for evaluation.
  uv run python pack.py  ->  $DATA/positions.npz, $DATA/human.npz"""
import json, math
from pathlib import Path
import numpy as np
L = Path(__file__).resolve().parent; T = L.parent
DATA = Path('/Volumes/Extreme SSD/Datasets/threes-v50')

def pack(rows):
    n = len(rows)
    out = {'board': np.zeros((n, 16), np.uint8), 'cards': np.zeros((n, 3), np.uint8), 'bonus': np.zeros(n, bool), 'counts': np.zeros((n, 3), np.uint8),
           'cycle': np.zeros((n, 3), np.uint8), 'values': np.full((n, 4), np.nan, np.float32), 'seed': np.zeros(n, np.int64), 'move': np.zeros(n, np.int32)}
    for i, r in enumerate(rows):
        out['board'][i] = r['board']; out['cards'][i, :len(r['cards'])] = r['cards']; out['bonus'][i] = r['bonus']
        out['counts'][i] = r['counts']; out['cycle'][i] = r['cycle']; out['values'][i] = r['values']; out['seed'][i] = r['seed']; out['move'][i] = r['move']
    return out

def margins_rows(files):
    for f in files:
        for line in f.open():
            m = json.loads(line)
            yield {'board': m['board'], 'cards': m['next']['cards'], 'bonus': m['next']['bonus'], 'counts': m['counts'], 'cycle': m['cycle'],
                   'values': [d['value'] if d['legal'] else math.nan for d in m['dirs']], 'seed': m['seed'], 'move': m['move']}

def rank(v): return v if v <= 3 else round(math.log2(v / 3)) + 3
def human_rows():
    """V47 (official app, cycle rule: public cycle state known) and V48 (PKU 2048, constant 1/21: no cycle;
    cycle fed as (0, 0, 0)). Bonus previews of V48 with several possible windows: union of the windows."""
    for src, p in [('v47', T / 'v47-human' / 'ai.jsonl')] + [('v48', p) for p in sorted((T / 'v48-logiccube' / 'ai').glob('*.jsonl'))]:
        for line in p.open():
            r = json.loads(line); nx = r['next']
            cards = sorted({rank(c) for w in nx.get('windows', [nx.get('candidates')]) for c in w})[:3]
            cy = r.get('cycle') or {}
            yield {'board': [rank(v) for row in r['board'] for v in row], 'cards': cards, 'bonus': nx['bonus'], 'counts': r['counts'],
                   'cycle': [cy.get('pos', 0), cy.get('used', 0), cy.get('excl', 0)] if isinstance(cy, dict) else list(cy),
                   'values': [math.nan if v is None else v for v in r['values']], 'seed': 47 if src == 'v47' else int(p.stem), 'move': r['i'], 'human': r['human']}

if __name__ == '__main__':
    files = sorted((T / 'v46-margins' / 'data').glob('m-*.jsonl')) + sorted((DATA / 'data').glob('m-*.jsonl'))
    d = pack(list(margins_rows(files)))
    d['val'] = (d['seed'] < 47000000) & (d['seed'] % 2 == 1)
    np.savez(DATA / 'positions.npz', **d)
    print(len(d['seed']), 'positions,', len(np.unique(d['seed'])), 'games; validation', int(d['val'].sum()))
    rows = list(human_rows()); h = pack(rows); h['human'] = np.array([r['human'] for r in rows], np.int8)
    np.savez(DATA / 'human.npz', **h)
    print(len(rows), 'human positions')
