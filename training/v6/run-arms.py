import subprocess,concurrent.futures,json,hashlib,gzip,datetime
from pathlib import Path
root=Path('training/v6');arms=['control','hard','probability']
def run(arm):
 with open(root/f'train-{arm}.log','w') as f:subprocess.run(['/tmp/threes-v6-train',str(root/f'run-{arm}'),arm,'450','202609217','dist/models/ntuple-v4.bin',str(root/'regular.txt'),str(root/'hard-pool.txt'),str(root/'teacher.txt')],stdout=f,stderr=subprocess.STDOUT,check=True)
 blob=(root/f'run-{arm}/checkpoint.ntd').read_bytes();base=Path('dist/models/ntuple-v4.bin').read_bytes();assert blob[16+8*65536*4:]==base[16+8*65536*4:]
 (root/f'candidate-{arm}.ntd.gz').write_bytes(gzip.compress(blob,mtime=0))
 with open(root/f'screen-{arm}.log','w') as f:subprocess.run(['python','training/v6/run-eval.py',str(root/f'run-{arm}/checkpoint.ntd'),str(int(arm=='probability')),'8310001','400',str(root/f'screen-{arm}.jsonl'),'2'],stdout=f,stderr=subprocess.STDOUT,check=True)
 return arm
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex: list(ex.map(run,arms))
results={}
for arm in arms:
 rows=[json.loads(x) for x in (root/f'screen-{arm}.jsonl').read_text().splitlines()];assert len(rows)==400 and not any(r['truncated'] for r in rows)
 results[arm]={str(t):sum(r['maxRank']>=rank for r in rows) for t,rank in [(1536,12),(3072,13),(6144,14)]}
selected=max(arms,key=lambda a:(results[a]['1536'],results[a]['6144'],-arms.index(a)))
blob=(root/f'run-{selected}/checkpoint.ntd').read_bytes();(root/'selected.ntd').write_bytes(blob)
(root/'selection.json').write_text(json.dumps({'selected':selected,'kind':'reach1536' if selected=='probability' else 'score','sha256':hashlib.sha256(blob).hexdigest(),'frozenAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'screen':results,'holdoutInspected':False},indent=2)+'\n')
print((root/'selection.json').read_text(),flush=True)
