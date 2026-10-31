"""V63 analysis (setup in RL-V63-EMPTY-SWITCH.md): arm K3 (5-ply / 0.01 at >= 3 empty cells, else 6-ply / 0.005)
against the V43 arm E records (6-ply / 0.005 every move), 4096 whole games paired by seed. Reach 6144 / 12288:
paired difference with a 95% CI and exact McNemar; non-inferior when both CI lower bounds >= -2pp. Timing: K3 vs
the arm E rerun on the same 256 seeds in the same job.  -> results.json"""
import json, math
from pathlib import Path
L = Path(__file__).resolve().parent; T = L.parent

def load(d, arm):
    return {r['seed']: r for f in sorted(d.glob(f'{arm}-*.jsonl')) for r in map(json.loads, f.read_text().splitlines())}

def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)

def paired(A, B, idx, key):                              # A - B
    d = [float(A[i][key]) - float(B[i][key]) for i in idx]; n = len(d); m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)); r = {'diff': m, 'ci95': [m - 1.96 * sd / math.sqrt(n), m + 1.96 * sd / math.sqrt(n)]}
    if isinstance(A[idx[0]][key], bool):
        b = sum(x == 1 for x in d); c = sum(x == -1 for x in d); r.update(onlyFirst=b, onlySecond=c, mcnemarP=mcnemar(b, c))
    return r

if __name__ == '__main__':
    K3, E = load(L / 'games', 'K3'), load(T / 'v43-early-depth' / 'games', 'E'); Er = load(L / 'games', 'E')
    idx = sorted(set(K3) & set(E)); out = {'games': len(idx), 'arms': {}}
    for k, A in (('E', E), ('K3', K3)):
        rs = [A[i] for i in idx]; n = len(rs)
        out['arms'][k] = {**{m: sum(r[m] for r in rs) / n for m in ('r1536', 'r3072', 'r6144', 'r12288', 'score', 'moves')},
                          'p12288given6144': sum(r['r12288'] for r in rs) / max(1, sum(r['r6144'] for r in rs))}
    out['arms']['K3']['cheapShare'] = sum(K3[i]['cheapMoves'] for i in idx) / sum(K3[i]['moves'] for i in idx)
    for m in ('r6144', 'r12288', 'r3072', 'score'): out[f'K3-E {m}'] = paired(K3, E, idx, m)
    out['nonInferior2pp'] = out['K3-E r6144']['ci95'][0] >= -0.02 and out['K3-E r12288']['ci95'][0] >= -0.02
    t = sorted(set(Er) & set(K3))
    out['timing'] = {'games': len(t), 'reproducesV43': all(Er[i][k] == E[i][k] for i in t for k in ('r6144', 'r12288', 'moves', 'score', 'turn6144')),
                     **{a: {'cpuPerGame': sum(A[i]['searchCPU'] for i in t) / len(t), 'msPerMove': 1000 * sum(A[i]['searchCPU'] for i in t) / sum(A[i]['moves'] for i in t)}
                        for a, A in (('E', Er), ('K3', K3))}}
    out['timing']['savedPerGame'] = 1 - out['timing']['K3']['cpuPerGame'] / out['timing']['E']['cpuPerGame']
    out['timing']['savedPerMove'] = 1 - out['timing']['K3']['msPerMove'] / out['timing']['E']['msPerMove']
    (L / 'results.json').write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out, indent=1))
