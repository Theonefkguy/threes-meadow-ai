import gzip,json
from pathlib import Path
root=Path(__file__).resolve().parents[2]
source=root/'training/run-v1-refined/snapshots.jsonl.gz'
target=root/'training/v2/initial-curriculum.txt'
with gzip.open(source,'rt') as src,target.open('w') as out:
    for line in src:
        row=json.loads(line)
        if row['stage']!=1: continue
        deck=row['deck']; counts=[deck.count(i) for i in [1,2,3]]
        values=row['ranks']+[len(deck)]+deck+[len(row['cards'])]
        for card,p in zip(row['cards'],row['weights']): values.extend([card,p])
        values.extend([row['turns']]+counts)
        out.write(' '.join(map(str,values))+'\n')
