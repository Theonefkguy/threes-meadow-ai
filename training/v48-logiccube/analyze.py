"""V48 analysis -> analysis.json: LogicCube's three PKU 2048 games against the website AI, per game and pooled.
Gaps and classes as in V47/V46: gap of a direction = (best root value - its value) / |best root value|.
Phases by the biggest card on the board; positions with a bonus preview are also reported apart,
since the 3-card window the player saw is unknown (values averaged over the possible windows)."""
import json, sys
from pathlib import Path
from collections import Counter
L = Path(__file__).resolve().parent
sys.path.insert(0, str(L.parent / 'v47-human'))
GAMES = ['11446', '6008', '10445']   # 12288 game, then the two 6144 games
DIRS = ['left', 'right', 'up', 'down']
CORNERS = {0, 3, 12, 15}; TOWARD = {0: {0, 2}, 3: {1, 2}, 12: {0, 3}, 15: {1, 3}}
PHASES = ('≤768', '1536', '3072', '6144')
CLASSES = ('clear', 'mixed', 'tie2', 'tie3', 'tie4')

def classify(g, k):
    if k == 1: return 'forced'
    if g[1] >= 0.02: return 'clear'
    if g[1] < 0.0025 and (k == 2 or g[2] >= 0.01): return 'tie2'
    if k >= 3 and g[2] < 0.0025 and (k == 3 or g[3] >= 0.01): return 'tie3'
    if k == 4 and g[3] < 0.0025: return 'tie4'
    return 'mixed'
def phase(mx): return '≤768' if mx <= 768 else '1536' if mx <= 1536 else '3072' if mx <= 3072 else '6144'

dec = []
for gid in GAMES:
    for line in (L / 'ai' / f'{gid}.jsonl').read_text().splitlines():
        r = json.loads(line); v = r['values']
        legal = [d for d in range(4) if v[d] is not None]; best = r['ai']; top = v[best]
        gaps = {d: (top - v[d]) / abs(top) if top else 0.0 for d in legal}
        b = [x for row in r['board'] for x in row]; mx = max(b); mc = max(range(16), key=lambda i: (b[i], -i))
        dec.append({'game': gid, 'i': r['i'], 't': r['t'], 'phase': phase(mx), 'k': len(legal), 'cls': classify(sorted(gaps.values()), len(legal)),
                    'human': r['human'], 'ai': best, 'agree': r['human'] == best, 'gap': gaps[r['human']], 'points': top - v[r['human']],
                    'corner': mc in CORNERS, 'humanToward': mc in CORNERS and r['human'] in TOWARD[mc], 'aiToward': mc in CORNERS and best in TOWARD[mc],
                    'bonusNext': r['next']['bonus'], 'windows': r['windows'], 'think': None if r['ms'] is None else r['ms'] / 1000,
                    'board': r['board'], 'next': r['next'], 'gaps': [gaps.get(d) for d in range(4)]})

def summ(xs):
    n = len(xs)
    if not n: return None
    g = sorted(x['gap'] for x in xs)
    return {'n': n, 'agree': sum(x['agree'] for x in xs) / n, 'meanGap': sum(g) / n, 'medianGap': g[n // 2],
            'within025': sum(x < 0.0025 for x in g) / n, 'over2': sum(x >= 0.02 for x in g) / n, 'over5': sum(x >= 0.05 for x in g) / n,
            'pointsLost': sum(x['points'] for x in xs)}
def corner(xs):
    cn = [x for x in xs if x['corner']]
    return {'n': len(cn), 'share': len(cn) / len(xs), 'humanToward': sum(x['humanToward'] for x in cn) / len(cn), 'aiToward': sum(x['aiToward'] for x in cn) / len(cn)} if cn else None
def med(a): a = sorted(a); return a[len(a) // 2] if a else None
def block(xs):
    nd = [x for x in xs if x['windows'] <= 1]
    th = [x for x in xs if x['think'] is not None and x['think'] < 60]
    return {'overall': summ(xs), 'singleWindow': summ(nd), 'bonusPreview': summ([x for x in xs if x['bonusNext']]),
            'byPhase': {p: summ([x for x in xs if x['phase'] == p]) for p in PHASES},
            'byClass': {c: summ([x for x in xs if x['cls'] == c]) for c in CLASSES},
            'byLegal': {k: summ([x for x in xs if x['k'] == k]) for k in (2, 3, 4)},
            'classByPhase': {p: dict(Counter(x['cls'] for x in xs if x['phase'] == p)) for p in PHASES},
            'corner': corner(xs), 'cornerByPhase': {p: corner([x for x in xs if x['phase'] == p]) for p in PHASES},
            'think': {c: {'n': len(s), 'median': med([x['think'] for x in s])} for c in CLASSES for s in [[x for x in th if x['cls'] == c]] if s},
            'thinkAgree': {str(a): {'n': len(s), 'median': med([x['think'] for x in s])} for a in (True, False) for s in [[x for x in th if x['agree'] == a]]},
            'thinkByPhase': {p: med([x['think'] for x in th if x['phase'] == p]) for p in PHASES}}

out = {'games': {}, 'pooled': block(dec)}
for gid in GAMES:
    g = json.loads((L / 'games' / f'{gid}.json').read_text())
    out['games'][gid] = {'score': g['score'], 'moves': g['moves'], 'highest': g['highest'], 'playedAt': g['playedAt'], 'elapsedS': g['elapsedMs'] / 1000,
                         'evaluated': sum(x['game'] == gid for x in dec), **block([x for x in dec if x['game'] == gid])}
v47 = json.loads((L.parent / 'v47-human' / 'analysis.json').read_text())
out['v47'] = {k: v47[k] for k in ('overall', 'byPhase', 'byClass', 'corner', 'think')}
out['worst'] = sorted(dec, key=lambda x: -x['gap'])[:15]
out['decisions'] = [{k: x[k] for k in ('game', 'i', 't', 'phase', 'k', 'cls', 'human', 'ai', 'agree', 'gap', 'points', 'think', 'bonusNext', 'windows')} for x in dec]
(L / 'analysis.json').write_text(json.dumps(out, ensure_ascii=False, separators=(',', ':')) + '\n')

def row(name, s): print(f"{name:14s} n={s['n']:5d} agree {s['agree']:.1%} ≤0.25% {s['within025']:.1%} ≥2% {s['over2']:.1%} ≥5% {s['over5']:.1%} meanGap {s['meanGap']:.2%}" if s else f'{name:14s} -')
for gid, g in [*out['games'].items(), ('pooled', out['pooled'])]:
    print('==', gid, {k: g[k] for k in ('score', 'moves', 'highest', 'evaluated') if k in g})
    row('overall', g['overall']); row('single window', g['singleWindow']); row('bonus preview', g['bonusPreview'])
    for p in PHASES: row(p, g['byPhase'][p])
    for c in CLASSES: row(c, g['byClass'][c])
    for p in PHASES: print(f"  corner {p}", g['cornerByPhase'][p] and {k: round(v, 3) for k, v in g['cornerByPhase'][p].items()})
    print('  think', g['think'], g['thinkAgree'], g['thinkByPhase'])
print('== V47 video player'); row('overall', out['v47']['overall'])
for k, s in out['v47']['byPhase'].items(): row(k, s)
for x in out['worst']: print(x['game'], x['i'], x['phase'], x['cls'], DIRS[x['human']], DIRS[x['ai']], f"{x['gap']:.1%}", x['windows'], x['board'], x['next'].get('candidates', x['next'].get('windows')))
