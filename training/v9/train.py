from pathlib import Path
import json,concurrent.futures,subprocess,hashlib,datetime
p=Path('training/v9');protocol=json.loads((p/'protocol.json').read_text());assert hashlib.sha256(Path(protocol['baseModel']).read_bytes()).hexdigest()==protocol['baseSHA256']
def run(job):
 arm,seed=job;out=p/f'{arm}-{seed}';subprocess.run(['/tmp/threes-v9-train',arm,str(seed),str(out)],check=True);print(arm,seed,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(run,[(a,s) for a in protocol['arms'] for s in protocol['trainingSeeds']]))
manifest={}
for a in protocol['arms']:
 for seed in protocol['trainingSeeds']:
  name=f'{a}-{seed}';out=p/name;rows=[json.loads(l) for l in (out/'metrics.jsonl').read_text().splitlines()];selected=[r for r in rows if r['selected']][-1];manifest[name]={'sha256':hashlib.sha256((out/'best.net').read_bytes()).hexdigest(),'checkpointEpoch':selected['epoch'],'validation':selected}
(p/'manifest.json').write_text(json.dumps({'frozenAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'models':manifest},indent=2)+'\n')
