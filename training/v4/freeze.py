from pathlib import Path
import json,hashlib,shutil
v=Path('training/v4');scores={}
for n in ['v3','control','a','b','c']:
 rows=[json.loads(l) for l in (v/f'screen-{n}.jsonl').read_text().splitlines()];assert len(rows)==400;assert [r['seed'] for r in rows]==list(range(6310001,6310401));assert not any(r['truncated'] for r in rows)
 scores[n]={'games':400,'wins6144':sum(r['success'] for r in rows),'reached3072':sum(r['maxRank']>=13 for r in rows)}
name=max(['control','a','c','b'],key=lambda n:scores[n]['wins6144']);src=v/f'run-{name}/checkpoint.ntd';shutil.copyfile(src,v/'selected.ntd');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();extra=None
if name=='b':shutil.copyfile(v/'run-b/checkpoint.five',v/'selected.five');extra=sha(v/'selected.five')
r={'winner':name,'screening':scores,'scoreSha256':sha(v/'selected.ntd'),'residualSha256':extra,'holdoutUsedForSelection':False,'testSeeds':[6410001,6415000]};(v/'selection.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
