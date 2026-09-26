from pathlib import Path
import json, random, gzip, subprocess, hashlib
p=Path('training/diagnosis-v7'); old=Path('training/v7')
indices=json.loads((old/'assessment-indices.json').read_text())
sources={r['index']:r for r in map(json.loads,(old/'sources.jsonl').read_text().splitlines()) if r['type']=='candidate'}
rows=[]
for line in (old/'search-teacher.txt').read_text().splitlines():
 x=line.split(); rows.append(dict(source=int(x[0]),direction=int(x[1]),weight=int(x[2]),target=float(x[3]),board=list(map(int,x[4:])),game=sources[indices[int(x[0])]]['seed']))
# Unite source games sharing an identical afterstate up to D4 symmetry.
def canon(b):
 versions=[]
 for reflect in range(2):
  for rot in range(4):
   c=[0]*16
   for pos,r in enumerate(b):
    y,x=divmod(pos,4)
    if reflect:x=3-x
    for _ in range(rot):y,x=x,3-y
    c[y*4+x]=r
   versions.append(tuple(c))
 return min(versions)
parent={r['game']:r['game'] for r in rows}
def find(a):
 while parent[a]!=a:a=parent[a]
 return a
seen={}
for r in rows:
 key=canon(r['board'])
 if key in seen:parent[find(r['game'])]=find(seen[key])
 else:seen[key]=r['game']
groups=sorted({find(g) for g in parent}); random.Random(202609220).shuffle(groups);held=set(groups[:round(len(groups)*.2)])
for r in rows:r['split']=int(find(r['game']) in held)
assert not ({canon(r['board']) for r in rows if r['split']} & {canon(r['board']) for r in rows if not r['split']})
(p/'fit-data.json').write_text(json.dumps(rows))
(p/'fit-data.txt').write_text(''.join(f"{r['source']} {r['direction']} {r['weight']} {r['split']} {r['target']} "+' '.join(map(str,r['board']))+'\n' for r in rows))
anchors=(old/'teacher.txt').read_text().splitlines(); random.Random(202609224).shuffle(anchors); (p/'normal-anchors.txt').write_text('\n'.join(anchors[:4096])+'\n')
(p/'fit-split.json').write_text(json.dumps({'groups':len(groups),'heldoutGroups':len(held),'examples':[sum(r['split']==s for r in rows) for s in range(2)],'groupBy':'collection game, union identical D4 afterstates','splitSeed':202609220},indent=2))
subprocess.run(['g++','-std=c++17','-O3','-march=native',str(p/'fit.cpp'),'-o','/tmp/threes-diagnosis-fit'],check=True)
def run(model,name,seed=202609221,epochs=0):
 subprocess.run(['/tmp/threes-diagnosis-fit',str(model),str(p/'fit-data.txt'),str(p/'normal-anchors.txt'),str(seed),str(epochs),str(p/(name+'.jsonl'))],check=True)
run(old/'base.ntd','fit-base')
for arm in ['r0','r10','r25','r10teacher']:
 model=Path('/tmp/threes-diagnosis-'+arm+'.ntd'); model.write_bytes(gzip.decompress((old/f'candidate-{arm}.ntd.gz').read_bytes())); run(model,'fit-'+arm)
for seed in [202609221,202609222,202609223]:run(old/'base.ntd',f'fit-pure-{seed}',seed,100)
print((p/'fit-split.json').read_text())
