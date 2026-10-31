"""V69 analysis: bench.txt (6000 positions, 4 arms) and games/*.jsonl (96 whole games, 3 arms) -> results.json."""
import glob
import json
import os
import statistics

here = os.path.dirname(os.path.abspath(__file__))
ARMS = ['original 1-6', 'fast 1-6', 'fast 1-4,6 (website)', 'fast 1,6']
rows = [l.split() for l in open(os.path.join(here, 'bench.txt'))]
arm = lambda r, a: r[1 + 8 * a:9 + 8 * a]  # ms nodes depth dir v0 v1 v2 v3
bench = {'positions': len(rows), 'arms': []}
base_ms = sum(float(arm(r, 0)[0]) for r in rows)
for a, name in enumerate(ARMS):
    ms = [float(arm(r, a)[0]) for r in rows]
    bench['arms'].append({
        'name': name,
        'sameDirectionAndValues': sum(arm(r, a)[3:] == arm(r, 0)[3:] for r in rows),
        'sameNodes': sum(arm(r, a)[1] == arm(r, 0)[1] for r in rows),
        'depth6': sum(arm(r, a)[2] == '6' for r in rows),
        'meanMs': round(statistics.mean(ms), 3),
        'p95Ms': round(sorted(ms)[int(0.95 * len(ms)) - 1], 2),
        'maxMs': round(max(ms), 2),
        'timeVsOriginal': round(sum(ms) / base_ms, 4),
        'nodesVsOriginal': round(sum(int(arm(r, a)[1]) for r in rows) / sum(int(arm(r, 0)[1]) for r in rows), 4),
    })

games = [json.loads(l) for f in sorted(glob.glob(os.path.join(here, 'games', '*.jsonl'))) for l in open(f)]
g = {'games': len(games), 'moves': sum(x['moves'] for x in games), 'arms': games[0]['arms'] if games else []}
if games:
    g['gamesWithAnyDifferentMove'] = [sum(x['differ'][a] > 0 for x in games) for a in range(len(g['arms']))]
    g['differentMoves'] = [sum(x['differ'][a] for x in games) for a in range(len(g['arms']))]
    g['secondsPerGame'] = [round(statistics.mean(x['ms'][a] for x in games) / 1000, 2) for a in range(len(g['arms']))]
    g['msPerMove'] = [round(sum(x['ms'][a] for x in games) / g['moves'], 3) for a in range(len(g['arms']))]
    g['timeVsOriginal'] = [round(sum(x['ms'][a] for x in games) / sum(x['ms'][0] for x in games), 4) for a in range(len(g['arms']))]
    ratios = [[x['ms'][a] / x['ms'][0] for x in games] for a in range(len(g['arms']))]
    g['perGameRatio'] = [{'median': round(statistics.median(r), 3), 'min': round(min(r), 3), 'max': round(max(r), 3)} for r in ratios]
    g['reach'] = {str(t): sum(x['high'] >= t for x in games) for t in (3072, 6144, 12288)}
    g['meanScore'] = round(statistics.mean(x['score'] for x in games))
    g['unsupported'] = sum(x['unsupported'] for x in games)
json.dump({'bench': bench, 'games': g}, open(os.path.join(here, 'results.json'), 'w'), indent=1)
print(json.dumps({'bench': bench, 'games': g}, indent=1))
