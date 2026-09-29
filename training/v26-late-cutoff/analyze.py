"""V26 analysis, fixed before outcomes (see analysis-pre-results-manifest.json).

python3 training/v26-late-cutoff/analyze.py   (repository root; needs games-*.jsonl,
browser-results.json and browser-reference.jsonl). Writes results.json.
"""
from pathlib import Path
import hashlib, json, math
from scipy import stats

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / 'protocol.json').read_text())
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
DIRS = ['left', 'right', 'up', 'down']

def wilson(k, n, z=1.959963984540054):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(c - h) / d, (c + h) / d]

def mcnemar(a, b):
    """Two-sided exact McNemar on paired booleans: a-only vs b-only."""
    ao = sum(x and not y for x, y in zip(a, b)); bo = sum(y and not x for x, y in zip(a, b))
    n = ao + bo
    p = 1.0 if n == 0 else min(1.0, 2 * float(stats.binom.cdf(min(ao, bo), n, 0.5)))
    return ao, bo, p

def paired(x, y):
    d = [a - b for a, b in zip(x, y)]; n = len(d); m = sum(d) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1)); se = sd / math.sqrt(n)
    t = m / se if se else 0.0
    return {'meanDifference': m, 'approx95': [m - 1.959963984540054 * se, m + 1.959963984540054 * se],
            'pairedT': t, 'pairedTp': float(2 * stats.t.sf(abs(t), n - 1)) if se else 1.0}

def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0.0] * len(ps); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i])); out[i] = running
    return out

def main():
    manifest = json.loads((HERE / 'pool-manifest.json').read_text())
    assert sha(HERE / 'pool6144.txt') == manifest['pool6144.txt']
    n = P['sampleSize']; arms = list(P['continuation']['arms'])
    games = {a: [json.loads(l) for l in (HERE / f'games-{a}.jsonl').read_text().splitlines()] for a in arms}
    seeds = [int(l.split()[0]) for l in (HERE / 'pool6144.txt').read_text().splitlines()]
    for a in arms:
        rows = games[a]; cfg = P['continuation']['arms'][a]
        assert [r['index'] for r in rows] == list(range(n)) and [r['sourceSeed'] for r in rows] == seeds
        assert all(r['depth'] == cfg['depth'] and abs(r['threshold'] - cfg['threshold']) < 1e-12 and r['startRank'] == 14 for r in rows)
        assert all(r['success'] == (r['finalRank'] >= 15) and (r['success'] or r['died']) for r in rows)
    win = {a: [r['success'] for r in games[a]] for a in arms}
    arm_stats = {a: {'reach12288': sum(win[a]), 'rate': sum(win[a]) / n, 'wilson95': wilson(sum(win[a]), n),
                     'meanScoreGained': sum(r['scoreGained'] for r in games[a]) / n,
                     'medianScoreGained': sorted(r['scoreGained'] for r in games[a])[n // 2],
                     'meanMoves': sum(r['moves'] for r in games[a]) / n,
                     'avoidableCertainDeath': sum(r['lastKind'] == 2 for r in games[a]),
                     'cpuMsPerMove': 1000 * sum(r['searchCPU'] for r in games[a]) / sum(r['moves'] for r in games[a]),
                     'maxMoveCpuMs': 1000 * max(r['maxMoveCPU'] for r in games[a])} for a in arms}
    primary = {}
    for pid, cand in (('P1', 'A1'), ('P2', 'A2')):
        ao, bo, p = mcnemar(win[cand], win['B'])
        primary[pid] = {'compare': f'{cand} vs B', 'candidateOnly': ao, 'baselineOnly': bo, 'difference': (sum(win[cand]) - sum(win['B'])) / n, 'mcnemarExactP': p}
    for pid, hp in zip(primary, holm([primary[k]['mcnemarExactP'] for k in primary])):
        primary[pid]['holmP'] = hp
    score = {f'{c}-B': paired([r['scoreGained'] for r in games[c]], [r['scoreGained'] for r in games['B']]) for c in ('A1', 'A2')}
    ao, bo, p = mcnemar(win['A2'], win['A1'])
    exploratory = {'A2 vs A1 reach12288': {'A2only': ao, 'A1only': bo, 'mcnemarExactP': p}}

    browser = json.loads((HERE / 'browser-results.json').read_text())
    native = [json.loads(l) for l in (HERE / 'browser-reference.jsonl').read_text().splitlines()]
    ref = {(r['index'], r['arm']): DIRS[r['direction']] for r in native}
    mismatches = [r for r in browser['parity'] if ref[(r['index'], r['arm'])] != r['direction']]
    assert len(browser['parity']) == len(native) == 128 * len(arms)
    bsum = {s['arm']: s for s in browser['summary']}
    browser_pass = {a: bsum[a]['completedFullDepth'] >= 0.99 * bsum[a]['n'] and bsum[a]['p95Ms'] < 160 for a in arms}

    gate = {}
    for pid, cand in (('P1', 'A1'), ('P2', 'A2')):
        pr = primary[pid]; sc = score[f'{cand}-B']
        gate[cand] = {'holmSignificantGain': pr['holmP'] < 0.05 and pr['candidateOnly'] > pr['baselineOnly'],
                      'scoreNotSignificantlyWorse': not (sc['pairedTp'] < 0.05 and sc['meanDifference'] < 0),
                      'browserPass': browser_pass[cand], 'parityPass': not mismatches}
        gate[cand]['qualifies'] = all(gate[cand].values())
    if gate['A1']['qualifies'] and gate['A2']['qualifies']:
        e = exploratory['A2 vs A1 reach12288']
        adopt = 'A2' if e['mcnemarExactP'] < 0.05 and e['A2only'] > e['A1only'] else 'A1'
    else:
        adopt = next((c for c in ('A1', 'A2') if gate[c]['qualifies']), None)

    result = {'protocolSHA256': sha(HERE / 'protocol.json'), 'pool': manifest, 'nPairs': n, 'arms': arm_stats, 'primary': primary,
              'scoreGained': score, 'exploratory': exploratory, 'browserSummary': browser['summary'], 'browserUserAgent': browser['userAgent'],
              'parityComparisons': len(native), 'parityMismatches': mismatches, 'adoptionGate': gate, 'adopt': adopt,
              'fileSHA256': {f: sha(HERE / f) for f in ['games-B.jsonl', 'games-A1.jsonl', 'games-A2.jsonl', 'browser-results.json', 'browser-reference.jsonl']}}
    (HERE / 'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'primary': primary, 'arms': {a: {k: arm_stats[a][k] for k in ('reach12288', 'meanScoreGained', 'cpuMsPerMove')} for a in arms},
                      'adopt': adopt}, indent=1))

if __name__ == '__main__':
    main()
