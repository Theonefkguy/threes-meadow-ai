import subprocess,concurrent.futures,sys,json,time
from pathlib import Path
model,goal,start,n,out,jobs=sys.argv[1:];start,n,jobs=int(start),int(n),int(jobs);out=Path(out);out.parent.mkdir(exist_ok=True,parents=True)
def worker(k):
 count=n//jobs+(k<n%jobs);first=start+(n//jobs)*k+min(k,n%jobs);part=Path(str(out)+f'.part{k}')
 with open(str(part)+'.log','w') as log:subprocess.run(['/tmp/threes-v3-benchmark-fast',model,goal,str(first),str(count),str(part),'0'],stdout=log,stderr=subprocess.STDOUT,check=True)
 return part
with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:parts=list(pool.map(worker,range(jobs)))
rows=[json.loads(l) for p in parts for l in p.read_text().splitlines()];assert len(rows)==n
out.write_text(''.join(json.dumps(r)+'\n' for r in rows));print(json.dumps({'file':str(out),'n':n,'wins':sum(r['success'] for r in rows),'truncated':sum(r['truncated'] for r in rows),'seconds':sum(r['seconds'] for r in rows)}),flush=True)
