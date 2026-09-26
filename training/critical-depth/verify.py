"""Audit a collected trajectory's state continuity and cut location."""
import json
from pathlib import Path
import sys

p=Path(sys.argv[1]); i=int(sys.argv[2]) if len(sys.argv)>2 else 0
rows=[json.loads(x) for x in (p/f'trajectory-{i}.jsonl').read_text().splitlines()]
meta=json.loads((p/f'source-{i}.json').read_text())
assert rows[-1]['over']
for k,x in enumerate(rows):
    assert x['action'] in x['legalActions']
    assert x['spawnRank'] in x['previewRanks']
    assert x['afterRanks'][x['spawnPosition']]==x['spawnRank']
    if k:
        assert rows[k-1]['afterRanks']==x['boardRanks']
        assert x['turn']==rows[k-1]['turn']+1
cut=rows[-meta['back']]
assert cut['turn']==meta['cutTurn']
assert meta['deathTurn']-meta['cutTurn']==meta['back']
assert cut['boardRanks']==list(map(int,(p/f'snapshot-{i}.txt').read_text().split()[:16]))
print(f'PASS: {len(rows)} transitions, cut {meta["back"]} steps before death')
