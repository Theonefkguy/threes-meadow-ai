from pathlib import Path
import json,random
p=Path('training/v15-late-depth')
lines=Path('training/v7/regular.txt').read_text().splitlines()
sources=[json.loads(x) for x in Path('training/v7/sources.jsonl').read_text().splitlines() if json.loads(x)['type']=='regular']
assert len(lines)==len(sources)==1993
rng=random.Random(202609281)
indices=sorted(rng.sample(range(len(lines)),128))
selected=[]
for new_index,old_index in enumerate(indices):
    fields=lines[old_index].split()
    board=list(map(int,fields[:16]))
    assert max(board)==11
    source=sources[old_index]
    assert source['index']==old_index
    selected.append({'index':new_index,'sourceIndex':old_index,'sourceSeed':source['seed'],'sourceTurn':source['turn']})
(p/'snapshots.txt').write_text('\n'.join(lines[i] for i in indices)+'\n')
(p/'snapshots.json').write_text(json.dumps(selected,indent=2)+'\n')
