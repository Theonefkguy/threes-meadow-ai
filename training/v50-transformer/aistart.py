"""V50 late starts: games that begin where the website AI first had a given card. Each AI game is
replayed in engine.py from its seed with the AI's recorded moves (margins 'chosen'); the board at the
start point is checked against the record. The student then plays on from there with the same RNG.
AI moves of all 1000 games are cached in $DATA/ai-moves.npz (built on first use).
  from aistart import late_games; late_games(seeds, rank) -> [(Game, top rank on the AI's last recorded board)]"""
import json
from pathlib import Path
import numpy as np
from engine import Game
L = Path(__file__).resolve().parent; DATA = Path('/Volumes/Extreme SSD/Datasets/threes-v50')
FILES = lambda: sorted((L.parent / 'v46-margins' / 'data').glob('m-*.jsonl')) + sorted((DATA / 'data').glob('m-*.jsonl'))
_cache = {}

def ai_moves():
    if not _cache:
        f = DATA / 'ai-moves.npz'
        if not f.exists():
            seeds, moves, boards = [], [], []
            for p in FILES():
                for line in p.open():
                    m = json.loads(line); seeds.append(m['seed']); moves.append(m['chosen']); boards.append(m['board'])
            np.savez(f, seed=np.array(seeds, np.int64), chosen=np.array(moves, np.int8), board=np.array(boards, np.uint8))
        d = np.load(f); s = d['seed']; cut = np.flatnonzero(np.diff(s)) + 1
        for a, b in zip(np.r_[0, cut], np.r_[cut, len(s)]): _cache[int(s[a])] = (d['chosen'][a:b], d['board'][a:b])
    return _cache

def ai_seeds(val):
    """Training games (V46 even seeds, V50) or validation games (V46 odd seeds), as in pack.py."""
    return sorted(s for s in ai_moves() if ((s < 47000000) and s % 2 == 1) == val)

def late_games(seeds, rank):
    out = []
    for s in seeds:
        chosen, boards = ai_moves()[s]; top = boards.max(1)
        k = np.flatnonzero(top >= rank)
        if not len(k): continue
        g = Game(s)
        for d in chosen[:k[0]]: g.step(int(d))
        assert list(g.b) == boards[k[0]].tolist(), f'replay mismatch seed {s}'
        out.append((g, int(top[-1])))
    return out
