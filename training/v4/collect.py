import subprocess,concurrent.futures,json
from pathlib import Path
v=Path('training/v4')
def run(k):
 p=v/f'fresh-pool.part{k}'
 with open(str(p)+'.log','w') as f:subprocess.run(['/tmp/threes-v4-collect','dist/models/ntuple-v3.bin',str(6110001+k*250),'250',str(p)],stdout=f,stderr=subprocess.STDOUT,check=True)
 return p
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:parts=list(ex.map(run,range(4)))
rows=[];sources=[]
for p in parts:
 start=len(rows);rows+=p.read_text().splitlines()
 for line in Path(str(p)+'.sources.jsonl').read_text().splitlines():
  d=json.loads(line);d['index']+=start;sources.append(d)
(v/'fresh-pool.txt').write_text('\n'.join(r.rstrip() for r in rows)+'\n');(v/'fresh-pool.sources.jsonl').write_text(''.join(json.dumps(d)+'\n' for d in sources));print(json.dumps({'normalGames':1000,'states':len(rows),'sourceGames':len(set(d['seed'] for d in sources))}),flush=True)
