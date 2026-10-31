"""V60 analysis: D7F (7-ply / 0.0025) against S6 (V40 S arm, 6-ply / 0.005) and D7 (V59, 7-ply / 0.005) on the
2048 first-6144 states, paired by state. Same statistics as V59 analyze.py.  -> results.json"""
import json, math, sys
from pathlib import Path
L = Path(__file__).resolve().parent; T = L.parent
sys.path.insert(0, str(T / 'v59-selective'))
from analyze import load, paired                                      # V59 helpers

arms = {'S6': load(T / 'v40-late-depth' / 'games', 'S'), 'D7': load(T / 'v59-selective' / 'games', 'D7'), 'D7F': load(L / 'games', 'D7F')}
idx = sorted(set.intersection(*(set(a) for a in arms.values())))
out = {'states': len(idx), 'arms': {}}
for k, A in arms.items():
    rs = [A[i] for i in idx]; mv = sum(r['moves'] for r in rs); cpu = sum(r['searchCPU'] for r in rs)
    p = sum(r['success'] for r in rs) / len(rs)
    out['arms'][k] = {'p12288': p, 'se': math.sqrt(p * (1 - p) / len(rs)), 'movesPerGame': mv / len(rs),
                      'scoreGained': sum(r['scoreGained'] for r in rs) / len(rs), 'cpuMsPerMove': 1000 * cpu / mv, 'cpuHours': cpu / 3600}
out['D7F-S6'] = paired(arms['D7F'], arms['S6'], idx); out['D7F-D7'] = paired(arms['D7F'], arms['D7'], idx)
out['D7FbetterThanS6'] = out['D7F-S6']['diff'] > 0 and out['D7F-S6']['mcnemarP'] < 0.05
(L / 'results.json').write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out, indent=1))
