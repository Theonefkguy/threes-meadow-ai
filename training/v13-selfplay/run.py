from pathlib import Path
import json,hashlib,subprocess,concurrent.futures,datetime
p=Path('training/v13-selfplay');pr=json.loads((p/'protocol.json').read_text())
for key in ['base','initial']:assert hashlib.sha256(Path(pr[key]).read_bytes()).hexdigest()==pr[key+'SHA256']
(p/'verification.txt').write_text(subprocess.check_output(['/tmp/threes-v13-verify',pr['initial']],text=True))
def evaljob(job):
 name,path,mode,start=job;subprocess.run(['/tmp/threes-v13-bench',path,mode,str(start),'16',str(p/f'eval-{name}.part{start}.jsonl')],check=True)
def evaluate(models):
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:list(ex.map(evaljob,[(n,path,mode,s) for s in range(11910001,11910513,16) for n,path,mode in models]))
 for n,_,_ in models:
  rows=sorted([json.loads(l) for f in p.glob(f'eval-{n}.part*.jsonl') for l in f.read_text().splitlines()],key=lambda x:x['seed']);assert [r['seed'] for r in rows]==list(range(11910001,11910513));(p/f'eval-{n}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
evaluate([('initial',pr['initial'],'actor'),('mcts','base','mcts')]);print('Initial student and MCTS evaluation complete',flush=True)
def train(seed):
 subprocess.run(['/tmp/threes-v13-train',pr['initial'],str(seed),str(p/f'run-{seed}')],check=True);print('Trained',seed,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(train,pr['trainingSeeds']))
manifest={};models=[]
for seed in pr['trainingSeeds']:
 for r in range(1,5):
  n=f'{seed}-round{r}';path=p/f'run-{seed}/round-{r}.net';manifest[n]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'episodes':2048*r};models.append((n,str(path),'actor'))
for key in ['base','initial']:assert hashlib.sha256(Path(pr[key]).read_bytes()).hexdigest()==pr[key+'SHA256']
(p/'manifest.json').write_text(json.dumps({'frozenAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'models':manifest},indent=2)+'\n');print('Final and intermediate checkpoints frozen',flush=True)
evaluate(models);print('Complete',flush=True)
