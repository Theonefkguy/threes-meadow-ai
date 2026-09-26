from pathlib import Path
import concurrent.futures,subprocess,json,hashlib,datetime,sys
p=Path('training/v16-adaptive-depth');pr=json.loads((p/'protocol.json').read_text())
if (p/'PAUSED').exists():
 print((p/'PAUSED').read_text().strip() or 'V16 paused')
 raise SystemExit(0)
assert hashlib.sha256(Path(pr['base']).read_bytes()).hexdigest()==pr['baseSHA256']
subprocess.run(['g++','-O3','-std=c++17',str(p/'continue.cpp'),'-o','/tmp/threes-v16-continue'],check=True)
jobs=[(i,r,m) for i in range(16) for r in range(2) for m in [0,5]]
def valid(path,job):
 try:
  x=json.loads(path.read_text());return (x['index'],x['replicate'],x['mode'])==job
 except Exception:return False
def run(job):
 i,r,m=job;out=p/f'result-i{i:03d}-r{r}-m{m}.json';progress=p/f'progress-i{i:03d}-r{r}-m{m}.jsonl'
 if valid(out,job):return 'cached'
 subprocess.run(['/tmp/threes-v16-continue','training/v15-late-depth/snapshots.txt',str(i),str(r),str(m),str(out),str(progress)],check=True)
 return 'done'
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:list(ex.map(run,jobs))
rows=[]
for job in jobs:
 i,r,m=job;path=p/f'result-i{i:03d}-r{r}-m{m}.json';assert valid(path,job);rows.append(json.loads(path.read_text()))
(p/'results-raw.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
(p/'status.json').write_text(json.dumps({'state':'complete','completed':len(rows),'total':len(jobs),'updatedAt':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
