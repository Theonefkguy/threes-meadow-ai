import subprocess,concurrent.futures,sys,json
from pathlib import Path
model,goal,out=sys.argv[1:];rows=Path('training/v3/late-eval.txt').read_text().splitlines();n=len(rows)
def run(k):
 p=out+f'.part{k}'
 subprocess.run(['/tmp/threes-v3-eval-late',model,goal,'training/v3/late-eval.txt',p,str(k*n//2),str((k+1)*n//2)],check=True)
 return Path(p).read_text()
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:parts=list(pool.map(run,range(2)))
Path(out).write_text(''.join(parts));data=[json.loads(l) for part in parts for l in part.splitlines()];print(json.dumps({'file':out,'states':n,'rollouts':len(data),'success':sum(r['success'] for r in data)}))
