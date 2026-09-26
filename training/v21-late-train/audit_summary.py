"""Summarise the death audit (run from repo root)."""
from pathlib import Path
import json
R = Path('training/v21-late-train'); out = {}
for e in ['E1', 'E2']:
    for m in ['CUR', 'NEW']:
        r = [json.loads(l) for f in sorted((R/'audit').glob(f'{e}-{m}-*.jsonl')) for l in f.read_text().splitlines()]
        if not r: continue
        died = [x for x in r if x['died']]
        out[f'{e}-{m}'] = {
            'games': len(r), 'success': sum(x['success'] for x in r), 'died': len(died),
            'deadOnArrivalSuccess': sum(x['success'] and x['overAtEnd'] for x in r),
            'finalMoveForcedCertainDeath': sum(x['lastKind'] == 1 for x in died),
            'finalMoveAvoidableCertainDeath': sum(x['lastKind'] == 2 for x in died),
            'finalMoveNotCertain': sum(x['lastKind'] == 0 for x in died),
            'gamesWithRiskierThanNeededMove': sum(x['riskyMoves'] > 0 for x in r),
            'riskierMoves': sum(x['riskyMoves'] for x in r),
            'riskierMovesWithCashInMerge': sum(x['riskyBigMerge'] for x in r),
            'expectedExtraDeaths': sum(x['riskyExtraP'] for x in r),
            'movesTotal': sum(x['moves'] for x in r)}
(R/'audit-summary.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
