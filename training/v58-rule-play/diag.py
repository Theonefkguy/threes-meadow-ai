"""V58 check (added after the whole-game results): how good are the rule players' moves on the positions
they reach themselves? 200 games per rule player (the first 200 seeds of play.py, same moves); every
position with >= 2 legal moves is scored by the website AI's 6-ply search (bin/label, 12 processes).
Reported: near (within 0.25% of the AI's best) and bad5 (>= 5% worse) of the player's moves, overall and
by the top card; and how often the last position with a choice in a game got a bad5 move.  -> diag.json"""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
L = Path(__file__).resolve().parent; T = L.parent; sys.path.insert(0, str(L))
from play import choose, PLAYERS, Game
from engine import move
Q = T / 'web-release' / 'models'; V50 = T / 'v50-transformer'
CMD = [str(V50 / 'bin' / 'label'), str(Q / 'ntuple-v31.bin'), ','.join(str(Q / f) for f in ('v31-s2q.res', 'v31-s1q.res', 'v31-s3q.res')), '-', '/dev/stdout']

def record(name, seed):
    g = Game(seed); rng = np.random.default_rng(seed); rows = []
    while not g.over and g.turns < 50000:
        d = choose(name, g, rng); nl = sum(bool(move(g.b, k)[1]) for k in range(4))
        if nl >= 2: rows.append((' '.join(map(str, [*g.b, len(g.next['cards']), *g.next['cards'], int(g.next['bonus']), *g.counts, *g.cycle])), d, max(g.b), seed))
        g.step(d)
    return rows

def label(lines):
    with tempfile.TemporaryDirectory() as tmp:                          # all 12 processes read their share at once
        fs = []
        for k in range(12):
            f = Path(tmp) / f'in{k}'; f.write_text(''.join(l + '\n' for l in lines[k::12])); fs.append(f.open())
        ws = [subprocess.Popen(CMD, stdin=f, stdout=subprocess.PIPE, text=True) for f in fs]
        vals = [w.communicate()[0].splitlines() for w in ws]
    res = [None] * len(lines)
    for k in range(12):
        for j, v in enumerate(vals[k]): res[k + 12 * j] = np.array([float(x) for x in v.split()])
    return np.array(res)

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200; out = {}
    for name in PLAYERS[1:]:
        rows = [r for s in range(n) for r in record(name, 50900001 + s)]
        V = label([r[0] for r in rows]); d = np.array([r[1] for r in rows]); top = np.array([r[2] for r in rows]); seed = np.array([r[3] for r in rows])
        best = np.nanmax(V, 1); gap = (best - V[np.arange(len(V)), d]) / np.abs(best)
        near, bad = gap < 0.0025, gap >= 0.05
        last = np.r_[seed[1:] != seed[:-1], True]
        by = {}
        for lo, hi, k in ((0, 6, '≤24'), (7, 7, '48'), (8, 8, '96'), (9, 99, '≥192')):   # ranks: 6 = 24, 7 = 48, 8 = 96
            m = (top >= lo) & (top <= hi)
            if m.sum() >= 100: by[k] = {'n': int(m.sum()), 'near': float(near[m].mean()), 'bad5': float(bad[m].mean())}
        out[name] = {'positions': len(rows), 'near': float(near.mean()), 'bad5': float(bad.mean()), 'byTop': by, 'lastMoveBad5': float(bad[last].mean()),
                     'bad5PerGame': float(bad.sum() / n)}
        print(name, json.dumps(out[name], ensure_ascii=False), flush=True)
        (L / 'diag.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
