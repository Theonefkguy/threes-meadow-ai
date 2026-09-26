from pathlib import Path
import json,hashlib,shutil,datetime
v=Path('training/v5');p=json.loads((v/'protocol.json').read_text());scores={}
assert not (v/'selection.json').exists(),'Selection already frozen'
for n in p['coefficients']:
 rows=[json.loads(l) for l in (v/f'screen-{n}.jsonl').read_text().splitlines()];assert len(rows)==400;assert [r['seed'] for r in rows]==list(range(7310001,7310401));assert not any(r['truncated'] for r in rows)
 scores[n]={'games':400,'wins6144':sum(r['success'] for r in rows),'reached3072':sum(r['maxRank']>=13 for r in rows),'lateMoves':sum(r['lateMoves'] for r in rows),'cornerMoves':sum(r['cornerMoves'] for r in rows)}
winner=max(p['coefficients'],key=lambda n:scores[n]['wins6144']);shaping=max(['weak','medium','strong'],key=lambda n:scores[n]['wins6144']);hashes={n:hashlib.sha256((v/f'run-{n}/checkpoint.ntd').read_bytes()).hexdigest() for n in p['coefficients']}
shutil.copyfile(v/f'run-{winner}/checkpoint.ntd',v/'selected.ntd')
r={'winner':winner,'bestShaping':shaping,'coefficient':p['coefficients'][winner],'screening':scores,'hashes':hashes,'frozenAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'holdoutUsedForSelection':False}
(v/'selection.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
