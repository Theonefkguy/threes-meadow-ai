"""V47 analysis -> analysis.json: the human game against the website AI's evaluation of every position.
Gap of a direction = (best root value - its value) / |best root value| (as in V46); classes as in V46."""
import json, sys
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'v46-margins'))
L = Path(__file__).resolve().parent
rows = [json.loads(l) for l in (L / 'ai.jsonl').read_text().splitlines()]
game = json.loads((L / 'game.json').read_text())
DIRS = ['left', 'right', 'up', 'down']
CORNERS = {0, 3, 12, 15}; TOWARD = {0: {0, 2}, 3: {1, 2}, 12: {0, 3}, 15: {1, 3}}

def classify(g, k):
    if k == 1: return 'forced'
    if g[1] >= 0.02: return 'clear'
    if g[1] < 0.0025 and (k == 2 or g[2] >= 0.01): return 'tie2'
    if k >= 3 and g[2] < 0.0025 and (k == 3 or g[3] >= 0.01): return 'tie3'
    if k == 4 and g[3] < 0.0025: return 'tie4'
    return 'mixed'
def phase(mx): return '1536' if mx <= 1536 else '3072' if mx <= 3072 else '6144'

dec = []
prev_t = None
for r in rows:
    v = r['values']; legal = [d for d in range(4) if v[d] is not None]; best = max(legal, key=lambda d: v[d]); top = v[best]
    gaps = {d: (top - v[d]) / abs(top) for d in legal}
    b = [x for row in r['board'] for x in row]; mx = max(b); mc = max(range(16), key=lambda i: (b[i], -i))
    rec = {'i': r['i'], 't': r['t'], 'phase': phase(mx), 'k': len(legal), 'cls': classify(sorted(gaps.values()), len(legal)),
           'human': r['human'], 'ai': best, 'agree': r['human'] == best, 'gap': gaps[r['human']], 'points': top - v[r['human']],
           'corner': mc in CORNERS, 'humanToward': mc in CORNERS and r['human'] in TOWARD[mc], 'aiToward': mc in CORNERS and best in TOWARD[mc],
           'bonusNext': r['next']['bonus'], 'think': None if prev_t is None else round(r['t'] - prev_t, 3), 'multi': r['multi'],
           'board': r['board'], 'next': r['next'], 'gaps': [None if d not in gaps else gaps[d] for d in range(4)]}
    prev_t = r['t']; dec.append(rec)

def summ(xs):
    n = len(xs)
    if not n: return None
    g = sorted(x['gap'] for x in xs)
    return {'n': n, 'agree': sum(x['agree'] for x in xs) / n, 'meanGap': sum(g) / n, 'medianGap': g[n // 2],
            'within025': sum(x < 0.0025 for x in g) / n, 'over2': sum(x >= 0.02 for x in g) / n, 'over5': sum(x >= 0.05 for x in g) / n,
            'pointsLost': sum(x['points'] for x in xs)}
out = {'moves': len(game['steps']), 'nDecisions': len(dec), 'durationS': round(game['steps'][-1]['t'] - game['steps'][0]['t'], 1),
       'overall': summ(dec), 'byPhase': {p: summ([x for x in dec if x['phase'] == p]) for p in ('1536', '3072', '6144')},
       'byClass': {c: summ([x for x in dec if x['cls'] == c]) for c in ('clear', 'mixed', 'tie2', 'tie3', 'tie4')},
       'byLegal': {k: summ([x for x in dec if x['k'] == k]) for k in (2, 3, 4)},
       'classByPhase': {p: dict(Counter(x['cls'] for x in dec if x['phase'] == p)) for p in ('1536', '3072', '6144')}}
cn = [x for x in dec if x['corner']]
out['corner'] = {'n': len(cn), 'share': len(cn) / len(dec), 'humanToward': sum(x['humanToward'] for x in cn) / len(cn), 'aiToward': sum(x['aiToward'] for x in cn) / len(cn),
                 'byPhase': {p: {'n': len(s), 'humanToward': sum(x['humanToward'] for x in s) / len(s), 'aiToward': sum(x['aiToward'] for x in s) / len(s)}
                             for p in ('1536', '3072', '6144') for s in [[x for x in cn if x['phase'] == p]] if s}}
th = [x for x in dec if x['think'] is not None and x['multi'] == 1 and x['think'] < 60]
def med(a): a = sorted(a); return a[len(a) // 2] if a else None
out['think'] = {c: {'n': len(s), 'median': med([x['think'] for x in s]), 'mean': sum(x['think'] for x in s) / len(s)} for c in ('clear', 'mixed', 'tie2', 'tie3', 'tie4') for s in [[x for x in th if x['cls'] == c]] if s}
out['thinkAgree'] = {str(a): {'n': len(s), 'median': med([x['think'] for x in s])} for a in (True, False) for s in [[x for x in th if x['agree'] == a]]}
out['worst'] = sorted(dec, key=lambda x: -x['gap'])[:12]
out['decisions'] = [{k: x[k] for k in ('i', 't', 'phase', 'k', 'cls', 'human', 'ai', 'agree', 'gap', 'points', 'think')} for x in dec]
(L / 'analysis.json').write_text(json.dumps(out, ensure_ascii=False, separators=(',', ':')) + '\n')
show = {k: out[k] for k in ('moves', 'nDecisions', 'durationS', 'overall', 'byPhase', 'byClass', 'byLegal', 'classByPhase', 'corner', 'think', 'thinkAgree')}
print(json.dumps(show, ensure_ascii=False, indent=1))
for x in out['worst']: print(x['i'], x['t'], x['phase'], x['cls'], DIRS[x['human']], DIRS[x['ai']], round(100 * x['gap'], 1), round(x['points']), x['board'], x['next']['candidates'])
