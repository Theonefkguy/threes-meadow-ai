"""V66 whole-game check (RL-V66-DIRECT-DEPTH.md): arms D6 (depth 6 only) and E (iterations from depth 1) on 256 V43
seeds against the V43 arm E records: every recorded field except the CPU time must match, and the first-6144
snapshots must match the V43 ones; CPU per game D6 vs E from the same job.  -> games.json"""
import json
from pathlib import Path
L = Path(__file__).resolve().parent; T = L.parent / 'v43-early-depth'
load = lambda d, a: {r['seed']: r for f in sorted(d.glob(f'{a}-*.jsonl')) for r in map(json.loads, f.read_text().splitlines())}
def snaps(d, a):
    out = {}
    for f in sorted(d.glob(f'{a}-*.txt')):
        for line in f.read_text().splitlines(): out[int(line.split()[0])] = line
    return out
V43, V43s = load(T / 'games', 'E'), snaps(T / 'snaps', 'E')
out = {}
keys = ('r1536', 'r3072', 'r6144', 'r12288', 'turn6144', 'c2', 'c3', 'c4', 'moves', 'score', 'bonuses', 'lateMoves')
arms = {a: (load(L / 'games', a), snaps(L / 'snaps', a)) for a in ('E', 'D6')}
seeds = sorted(arms['D6'][0]); assert len(seeds) == 256 and set(seeds) == set(arms['E'][0])
for a, (G, S) in arms.items():
    out[a] = {'games': len(G), 'sameRecords': sum(all(G[s][k] == V43[s][k] for k in keys) for s in seeds),
              'snapshots': len(S), 'sameSnapshots': sum(S[s] == V43s.get(s) for s in S),
              'cpuPerGame': sum(G[s]['searchCPU'] for s in seeds) / len(seeds),
              'msPerMove': 1000 * sum(G[s]['searchCPU'] for s in seeds) / sum(G[s]['moves'] for s in seeds)}
r = [arms['D6'][0][s]['searchCPU'] / arms['E'][0][s]['searchCPU'] for s in seeds]
out['D6/E cpu'] = out['D6']['cpuPerGame'] / out['E']['cpuPerGame']
out['D6/E per game: min, median, max'] = [min(r), sorted(r)[len(r) // 2], max(r)]
(L / 'games.json').write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out, indent=1))
