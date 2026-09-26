import json,hashlib,shutil
from pathlib import Path
v=Path('training/v3');results={}
for name in ['v2','a','b','c']:
 rows=[json.loads(l) for l in (v/f'screen-{name}.jsonl').read_text().splitlines()]
 assert len(rows)==200 and sorted(r['seed'] for r in rows)==list(range(2910001,2910201))
 assert not any(r['truncated'] for r in rows)
 results[name]={'games':200,'wins':sum(r['success'] for r in rows),'reached3072':sum(r['maxRank']>=13 for r in rows)}
winner=max(['a','b','c'],key=lambda n:results[n]['wins'])
score=v/('run-c/checkpoint.ntd' if winner=='c' else 'a.ntd');goal=v/'run-b/checkpoint.goal' if winner=='b' else None
shutil.copyfile(score,v/'selected.ntd')
if goal:shutil.copyfile(goal,v/'selected.goal')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
selection={'winner':winner,'screening':results,'scoreSha256':sha(v/'selected.ntd'),'goalSha256':sha(v/'selected.goal') if goal else None,'holdoutUsedForSelection':False,'holdoutSeeds':[3910001,3915000]}
(v/'selection.json').write_text(json.dumps(selection,indent=2)+'\n');print(json.dumps(selection))
