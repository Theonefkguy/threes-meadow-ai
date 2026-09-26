from pathlib import Path
import json,hashlib,subprocess,concurrent.futures,datetime,gzip
p=Path('training/v10');pr=json.loads((p/'protocol.json').read_text());base=Path(pr['base']);assert hashlib.sha256(base.read_bytes()).hexdigest()==pr['baseSHA256']
def collect(i):
 subprocess.run(['/tmp/threes-v10-collect',str(base),str(11210001+i*16),'16',str(p/f'pool.part{i}')],check=True,stdout=subprocess.DEVNULL)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:list(ex.map(collect,range(8)))
for suffix in ['regular.txt','games.jsonl']:
 (p/f'pool-{suffix}').write_text(''.join((p/f'pool.part{i}.{suffix}').read_text() for i in range(8)))
sources=[]
for i in range(8):
 for line in (p/f'pool.part{i}.sources.jsonl').read_text().splitlines():
  row=json.loads(line)
  if row['type']=='regular':row['index']=len(sources);sources.append(row)
(p/'pool-sources.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in sources))
print('Pool complete',flush=True)
def train(job):
 a,s=job;subprocess.run(['/tmp/threes-v10-train',str(p/f'{a}-{s}'),str(pr['arms'][a]),str(pr['cpuSecondsPerRun']),str(s),str(base),str(p/'pool-regular.txt')],check=True);print('Trained',a,s,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:list(ex.map(train,[(a,s) for a in pr['arms'] for s in pr['trainingSeeds']]))
manifest={};baseline=base.read_bytes();boundary=16+8*65536*4
for a in pr['arms']:
 for s in pr['trainingSeeds']:
  name=f'{a}-{s}';file=p/name/'checkpoint.ntd';data=file.read_bytes();assert len(data)==len(baseline) and data[boundary:]==baseline[boundary:];assert data[:boundary]!=baseline[:boundary]
  rows=[json.loads(l) for l in (p/name/'progress.jsonl').read_text().splitlines()];assert rows[-1]['truncated']==0
  with gzip.open(str(file)+'.gz','wb') as f:f.write(data)
  manifest[name]={'sha256':hashlib.sha256(data).hexdigest(),'lateStagesUnchanged':True,'training':rows[-1]}
(p/'manifest.json').write_text(json.dumps({'frozenAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'models':manifest},indent=2)+'\n');print('Manifest frozen',flush=True)
names=['base']+list(manifest)
def evaluate(job):
 name,start=job;model=base if name=='base' else p/name/'checkpoint.ntd';out=p/f'eval-{name}.part{start}.jsonl'
 subprocess.run(['/tmp/threes-v10-bench',str(model),'mcts',str(start),'16','1',str(out)],check=True);print('Evaluated',name,start,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:list(ex.map(evaluate,[(name,s) for s in range(11310001,11310513,16) for name in names]))
for name in names:
 rows=sorted([json.loads(l) for f in p.glob(f'eval-{name}.part*.jsonl') for l in f.read_text().splitlines()],key=lambda x:x['seed']);assert [r['seed'] for r in rows]==list(range(11310001,11310513))
 (p/f'eval-{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
print('Complete',flush=True)
