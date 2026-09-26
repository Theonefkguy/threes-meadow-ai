from pathlib import Path
import concurrent.futures,subprocess,json,hashlib,datetime,sys
p=Path('training/v15-late-depth');pr=json.loads((p/'protocol.json').read_text())
if (p/'PAUSED').exists():
 print((p/'PAUSED').read_text().strip() or 'V15 paused')
 raise SystemExit(0)
assert hashlib.sha256(Path(pr['base']).read_bytes()).hexdigest()==pr['baseSHA256']
subprocess.run(['g++','-O3','-std=c++17',str(p/'continue.cpp'),'-o','/tmp/threes-v15-continue'],check=True)
subprocess.run([sys.executable,str(p/'prepare.py')],check=True)
jobs=[(i,r,d) for i in range(128) for r in range(4) for d in [4,5]]
def valid(path,job):
 try:
  x=json.loads(path.read_text());return (x['index'],x['replicate'],x['depth'])==job
 except Exception:return False
def run(job):
 i,r,d=job;out=p/f'result-i{i:03d}-r{r}-d{d}.json';progress=p/f'progress-i{i:03d}-r{r}-d{d}.jsonl'
 if valid(out,job):return 'cached'
 subprocess.run(['/tmp/threes-v15-continue',str(p/'snapshots.txt'),str(i),str(r),str(d),str(out),str(progress)],check=True)
 return 'done'
completed=0
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
 for result in ex.map(run,jobs):
  completed+=1
  if completed%16==0:
   (p/'status.json').write_text(json.dumps({'state':'running','completed':completed,'total':len(jobs),'updatedAt':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
rows=[]
for job in jobs:
 i,r,d=job;path=p/f'result-i{i:03d}-r{r}-d{d}.json';assert valid(path,job);rows.append(json.loads(path.read_text()))
(p/'results-raw.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
assert hashlib.sha256(Path(pr['base']).read_bytes()).hexdigest()==pr['baseSHA256']
subprocess.run([sys.executable,str(p/'analyze.py')],check=True)
subprocess.run([sys.executable,str(p/'report.py')],check=True)
(p/'status.json').write_text(json.dumps({'state':'complete','completed':len(jobs),'total':len(jobs),'updatedAt':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
