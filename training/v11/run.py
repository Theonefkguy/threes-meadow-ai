from pathlib import Path
import json,hashlib,subprocess,concurrent.futures,datetime
p=Path('training/v11');pr=json.loads((p/'protocol.json').read_text());assert hashlib.sha256(Path(pr['base']).read_bytes()).hexdigest()==pr['baseSHA256']
def collect(i):
 subprocess.run(['/tmp/threes-v11-collect',str(11410001+i*16),'16',str(p/f'collection.part{i}')],check=True);print('Collected',i,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:list(ex.map(collect,range(32)))
for target,suffix in [('teacher.txt','txt'),('collection-games.jsonl','games.jsonl')]:
 (p/target).write_text(''.join((p/f'collection.part{i}.{suffix}').read_text() for i in range(32)))
print('Collection complete',flush=True)
def train(seed):
 subprocess.run(['/tmp/threes-v11-train','context_joint',str(seed),str(p/f'policy-{seed}')],check=True);print('Trained',seed,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(train,pr['trainingSeeds']))
manifest={}
for s in pr['trainingSeeds']:
 name=f'policy-{s}';out=p/name;rows=[json.loads(l) for l in (out/'metrics.jsonl').read_text().splitlines()];chosen=[r for r in rows if r['selected']][-1]
 verification=subprocess.check_output(['/tmp/threes-v11-verify',str(out/'best.net')],text=True);(out/'verification.txt').write_text(verification)
 manifest[name]={'sha256':hashlib.sha256((out/'best.net').read_bytes()).hexdigest(),'epoch':chosen['epoch'],'metrics':chosen}
assert hashlib.sha256(Path(pr['base']).read_bytes()).hexdigest()==pr['baseSHA256']
(p/'manifest.json').write_text(json.dumps({'frozenAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'models':manifest},indent=2)+'\n');print('Manifest frozen',flush=True)
names=['base']+list(manifest)
def evaluate(job):
 name,start=job;subprocess.run(['/tmp/threes-v11-bench','base' if name=='base' else str(p/name/'best.net'),name,str(start),'16','1',str(p/f'eval-{name}.part{start}.jsonl')],check=True);print('Evaluated',name,start,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:list(ex.map(evaluate,[(n,s) for s in range(11510001,11510513,16) for n in names]))
for n in names:
 rows=sorted([json.loads(l) for f in p.glob(f'eval-{n}.part*.jsonl') for l in f.read_text().splitlines()],key=lambda x:x['seed']);assert [r['seed'] for r in rows]==list(range(11510001,11510513));(p/f'eval-{n}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
print('Complete',flush=True)
