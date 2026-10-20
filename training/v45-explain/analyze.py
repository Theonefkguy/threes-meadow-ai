"""V45 analysis -> results.json: scan statistics and the counterfactual test (AI move vs the best calm move)."""
import json, math
from pathlib import Path
L = Path('training/v45-explain')
games = [json.loads(l) for f in (L / 'scan').glob('g-*.jsonl') for l in f.read_text().splitlines()]
moves = [json.loads(l) for f in (L / 'scan').glob('m-*.jsonl') for l in f.read_text().splitlines()]
total = sum(g['moves'] for g in games)
scan = {'games': len(games), 'moves': total, 'P12288': sum(g['success'] for g in games) / len(games),
        'rearrangingPerMove': sum(g['rearranging'] for g in games) / total, 'cornerLeavePerMove': sum(g['cornerLeave'] for g in games) / total}
samp = [m for m in moves if m['sample']]
def margin(m):
    v = sorted((m['values'][d] for d in range(4) if m['legal'][d]), reverse=True); return (v[0] - v[1]) / abs(v[0]) if len(v) > 1 else None
ms = sorted(x for x in map(margin, samp) if x is not None); scan['baselineMedianRelativeMargin'] = ms[len(ms) // 2]
re = sorted(margin(m) for m in moves if m['rearranging'] and margin(m) is not None); scan['rearrangingMedianRelativeMargin'] = re[len(re) // 2]
jobs = {j['position']: j for j in map(json.loads, (L / 'jobs.jsonl').read_text().splitlines())}
rows = [json.loads(l) for f in (L / 'rollouts').glob('r-*.jsonl') for l in f.read_text().splitlines()]
per = {}
for r in rows: per.setdefault(r['position'], {}).setdefault(r['arm'], []).append(r)
res = {'scan': scan, 'selection': json.loads((L / 'selection.json').read_text()), 'tests': {}}
for name in ('R', 'C'):
    ps = [p for p in per if jobs[p]['set'] == name]
    d = [sum(x['success'] for x in per[p]['ai']) / len(per[p]['ai']) - sum(x['success'] for x in per[p]['calm']) / len(per[p]['calm']) for p in ps]
    sd = [sum(x['scoreGained'] for x in per[p]['ai']) / len(per[p]['ai']) - sum(x['scoreGained'] for x in per[p]['calm']) / len(per[p]['calm']) for p in ps]
    def stat(v):
        n = len(v); m = sum(v) / n; se = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) / math.sqrt(n)
        return {'n': n, 'mean': m, 'approx95': [m - 1.96 * se, m + 1.96 * se], 'p': math.erfc(abs(m / se) / math.sqrt(2)) if se else 1.0}
    rate = lambda arm: sum(x['success'] for p in ps for x in per[p][arm]) / sum(len(per[p][arm]) for p in ps)
    res['tests'][name] = {'positions': len(ps), 'P12288 ai': rate('ai'), 'P12288 calm': rate('calm'), 'difference ai - calm': stat(d), 'score ai - calm': stat(sd)}
ks = sorted(res['tests'], key=lambda k: res['tests'][k]['difference ai - calm']['p']); run = 0
for i, k in enumerate(ks): run = max(run, min(1.0, (2 - i) * res['tests'][k]['difference ai - calm']['p'])); res['tests'][k]['holmP'] = run
(L / 'results.json').write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))
