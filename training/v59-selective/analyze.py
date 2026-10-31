"""V59 analysis: S6 (V40 S arm records), D7, SEL on the 2048 first-6144 states, paired by state.
P(12288), paired SEL - D7 and D7 - S6 differences (95% CI from the per-state differences, exact McNemar),
CPU ms per move, share of deepened moves, score gained.  -> results.json"""
import json, math
from pathlib import Path
L = Path(__file__).resolve().parent; T = L.parent

def load(d, arm):
    rows = {}
    for f in sorted(d.glob(f'{arm}-*.jsonl')):
        for line in f.read_text().splitlines(): r = json.loads(line); rows[r['index']] = r
    return rows

def mcnemar(b, c):                                       # exact two-sided binomial test on the discordant pairs
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)

def paired(A, B, idx):                                   # A - B
    d = [int(A[i]['success']) - int(B[i]['success']) for i in idx]; n = len(d); m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)); b = sum(x == 1 for x in d); c = sum(x == -1 for x in d)
    return {'diff': m, 'ci95': [m - 1.96 * sd / math.sqrt(n), m + 1.96 * sd / math.sqrt(n)], 'onlyFirst': b, 'onlySecond': c, 'mcnemarP': mcnemar(b, c)}

if __name__ == '__main__':
    arms = {'S6': load(T / 'v40-late-depth' / 'games', 'S'), 'D7': load(L / 'games', 'D7'), 'SEL': load(L / 'games', 'SEL')}
    idx = sorted(set.intersection(*(set(a) for a in arms.values())))
    out = {'states': len(idx), 'arms': {}}
    for k, A in arms.items():
        rs = [A[i] for i in idx]; mv = sum(r['moves'] for r in rs); cpu = sum(r['searchCPU'] for r in rs)
        p = sum(r['success'] for r in rs) / len(rs)
        out['arms'][k] = {'p12288': p, 'se': math.sqrt(p * (1 - p) / len(rs)), 'movesPerGame': mv / len(rs),
                          'scoreGained': sum(r['scoreGained'] for r in rs) / len(rs), 'cpuMsPerMove': 1000 * cpu / mv,
                          'cpuHours': cpu / 3600, 'deepShare': sum(r['deepMoves'] for r in rs) / mv}
    out['SEL-D7'] = paired(arms['SEL'], arms['D7'], idx); out['D7-S6'] = paired(arms['D7'], arms['S6'], idx); out['SEL-S6'] = paired(arms['SEL'], arms['S6'], idx)
    out['SELnoninferior2pp'] = out['SEL-D7']['ci95'][0] >= -0.02
    (L / 'results.json').write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out, indent=1))
