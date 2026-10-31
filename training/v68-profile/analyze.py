"""V68 bench analysis (RL-V68-PROFILE.md): per arm, positions whose move, node count and four root values are
bit-identical to arm 0 (original search and model, depth 6 only), and CPU time relative to arm 0.  -> results.json"""
import json
from pathlib import Path
L = Path(__file__).resolve().parent
names = ['original', 'maps', 'maps+half', 'neon', 'neon+half']
rows = [l.split() for l in (L / 'bench.txt').read_text().splitlines()]
assert len(rows) == 6000 and all(int(r[0]) == i for i, r in enumerate(rows))
t0 = sum(float(r[1]) for r in rows)
out = {'positions': len(rows), 'msOriginal': t0 / len(rows), 'arms': {}}
for j, n in enumerate(names):
    c = slice(1 + 7 * j, 8 + 7 * j)
    same = sum(r[c][1:] == r[1:8][1:] for r in rows)        # nodes, direction and the four values as printed (%.17g)
    t = sum(float(r[1 + 7 * j]) for r in rows)
    out['arms'][n] = {'identical': same, 'msPerMove': t / len(rows), 'msShare': t / t0}
(L / 'results.json').write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out, indent=1))
