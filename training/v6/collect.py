import subprocess, concurrent.futures, json
from pathlib import Path
root=Path('training/v6'); base='dist/models/ntuple-v4.bin'; jobs=8

def collect(k):
 p=root/f'collection.part{k}'
 with open(str(p)+'.log','w') as f: subprocess.run(['/tmp/threes-v6-collect',base,str(8110001+k*250),'250',str(p)],stdout=f,stderr=subprocess.STDOUT,check=True)
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
n=offsets['candidate']
def assess(k):
 count=n//jobs+(k<n%jobs); first=(n//jobs)*k+min(k,n%jobs); p=root/f'assessment.part{k}'
 with open(str(p)+'.log','w') as f: subprocess.run(['/tmp/threes-v6-assess',base,str(root/'candidates.txt'),str(first),str(count),str(p)],stdout=f,stderr=subprocess.STDOUT,check=True)
 return p
with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex: parts=list(ex.map(assess,range(jobs)))
rows=[json.loads(x) for p in parts for x in p.read_text().splitlines()]
assert len(rows)==n and [r['index'] for r in rows]==list(range(n))
(root/'assessment.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
selected=[r['index'] for r in rows if 0<r['wins']<r['trials']]
candidates=(root/'candidates.txt').read_text().splitlines()
(root/'hard-pool.txt').write_text(''.join(candidates[i]+'\n' for i in selected))
(root/'hard-indices.json').write_text(json.dumps(selected))
assert selected
print(json.dumps({'hardStates':len(selected),'candidates':n}),flush=True)
