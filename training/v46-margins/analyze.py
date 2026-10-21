"""V46: how close are the website AI's options? -> results.json (descriptive)."""
import csv, glob, json, statistics as st
from pathlib import Path
L = Path('training/v46-margins')
rows = [r for f in sorted(glob.glob(str(L / 'data/m-*.csv'))) for r in csv.reader(open(f))]
def bucket(rank):
    return '≤192' if rank <= 9 else '384–768' if rank <= 11 else {12: '1536', 13: '3072'}.get(rank, '6144')
B = {}
games = {r[0] for r in rows}
for r in rows:
    rank, legal = int(r[2]), int(r[3]); vals = sorted((float(x) for x in r[4:8] if x != ''), reverse=True)
    b = B.setdefault(bucket(rank), {'n': 0, 'forced': 0, 'g2': [], 'g3': [], 'abs2': [], 'cv': [], 'legal': []})
    b['n'] += 1; b['legal'].append(legal)
    if legal < 2: b['forced'] += 1; continue
    top = max(abs(vals[0]), 1.0)
    b['g2'].append((vals[0] - vals[1]) / top); b['abs2'].append(vals[0] - vals[1])
    if legal >= 3: b['g3'].append((vals[0] - vals[2]) / top)
    m = sum(vals) / len(vals); b['cv'].append(st.pstdev(vals) / top)
order = ['≤192', '384–768', '1536', '3072', '6144']
res = {'games': len(games), 'decisions': len(rows), 'phases': {}}
for k in order:
    b = B.get(k)
    if not b: continue
    g2, g3 = sorted(b['g2']), sorted(b['g3']); n2 = len(g2)
    share = lambda xs, t: sum(x < t for x in xs) / len(xs) if xs else None
    res['phases'][k] = {'decisions': b['n'], 'shareOfAll': b['n'] / len(rows), 'meanLegal': sum(b['legal']) / b['n'], 'onlyOneMove': b['forced'] / b['n'],
        'medianGapTop2': g2[n2 // 2], 'meanGapTop2': sum(g2) / n2, 'medianGapTop2Points': sorted(b['abs2'])[n2 // 2],
        'top2Within': {t: share(g2, t / 100) for t in (0.1, 0.25, 0.5, 1)},
        'top3Within': {t: share(g3, t / 100) for t in (0.1, 0.25, 0.5, 1)},
        'gapAbove5pct': sum(x >= 0.05 for x in g2) / n2,
        'meanRelativeStd': sum(b['cv']) / len(b['cv'])}
(L / 'results.json').write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n'); print(json.dumps(res, indent=1, ensure_ascii=False))
