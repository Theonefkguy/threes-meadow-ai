import subprocess, concurrent.futures, json
from pathlib import Path
root=Path('training/v7'); base='training/v7/base.ntd'; jobs=8

def collect(k):
 p=root/f'collection.part{k}'
 with open(str(p)+'.log','w') as f: subprocess.run(['/tmp/threes-v7-collect',base,str(9110001+k*250),'250',str(p)],stdout=f,stderr=subprocess.STDOUT,check=True)
 return p
with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex: parts=list(ex.map(collect,range(jobs)))
offsets={'regular':0,'candidate':0}; sources=[]
for ext in ['regular.txt','candidates.txt','teacher.txt','games.jsonl']:
 (root/ext).write_text(''.join(line.rstrip()+'\n' for p in parts for line in Path(str(p)+'.'+ext).read_text().splitlines()))
for p in parts:
 rows=[json.loads(x) for x in Path(str(p)+'.sources.jsonl').read_text().splitlines()]
 for r in rows: r['index']+=offsets[r['type']]; sources.append(r)
 for kind,ext in [('regular','regular.txt'),('candidate','candidates.txt')]: offsets[kind]+=len(Path(str(p)+'.'+ext).read_text().splitlines())
(root/'sources.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in sources))
print(json.dumps({'collection':offsets}),flush=True)
