"""V45 step 2 selection (rule in protocol.json): one random rearranging move per game for set R and one
random cornerLeave-only move per game for set C, 120 games each (random.Random(45)); writes jobs.jsonl."""
import json, random
from pathlib import Path
L = Path('training/v45-explain')
moves = [json.loads(l) for f in sorted((L / 'scan').glob('m-*.jsonl')) for l in f.read_text().splitlines()]
moves.sort(key=lambda r: (r['index'], r['move']))
rng = random.Random(45); jobs = []; summary = {}
for name, ok, alt in (('R', lambda r: r['rearranging'], lambda r, d: r['bigMoved'][d] == 0),
                      ('C', lambda r: r['cornerLeave'] and not r['rearranging'], lambda r, d: r['keepsCorner'][d] == 1)):
    by = {}
    for r in moves:
        if ok(r): by.setdefault(r['index'], []).append(r)
    games = sorted(by); rng.shuffle(games); pick = games[:120]
    summary[name] = {'eligibleMoves': sum(len(v) for v in by.values()), 'eligibleGames': len(games), 'picked': len(pick)}
    for g in pick:
        r = rng.choice(by[g]); a = max((d for d in range(4) if r['legal'][d] and alt(r, d)), key=lambda d: r['values'][d])
        jobs.append({'position': len(jobs), 'set': name, 'index': r['index'], 'move': r['move'], 'aiValue': r['values'][r['chosen']], 'altValue': r['values'][a],
                     'state': r['state'], 'arms': {'ai': r['chosen'], 'calm': a}})
(L / 'jobs.jsonl').write_text(''.join(json.dumps(j, separators=(',', ':')) + '\n' for j in jobs))
(L / 'selection.json').write_text(json.dumps(summary, indent=1) + '\n'); print(summary, len(jobs))
