from pathlib import Path
import random,json,subprocess,concurrent.futures
root=Path('training/v7');lines=(root/'candidates.txt').read_text().splitlines();indices=sorted(random.Random(202609219).sample(range(len(lines)),min(256,len(lines))));(root/'assessment-indices.json').write_text(json.dumps(indices)+'\n');(root/'assessment-pool.txt').write_text(''.join(lines[i]+'\n' for i in indices));n=len(indices);jobs=8

def run(k):
 count=n//jobs+(k<n%jobs);first=n//jobs*k+min(k,n%jobs);prefix=root/f'assessment.part{k}'
 with open(str(prefix)+'.log','w') as f:subprocess.run(['/tmp/threes-v7-assess',str(root/'base.ntd'),str(root/'assessment-pool.txt'),str(first),str(count),str(prefix)],stdout=f,stderr=subprocess.STDOUT,check=True)
 return prefix
with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:parts=list(ex.map(run,range(jobs)))
rows=[json.loads(l) for p in parts for l in Path(str(p)+'.jsonl').read_text().splitlines()];assert [r['index'] for r in rows]==list(range(n));(root/'assessment.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));hard=[indices[r['index']] for r in rows if r['hard']];assert hard;(root/'hard-indices.json').write_text(json.dumps(hard)+'\n');(root/'hard-pool.txt').write_text(''.join(lines[i]+'\n' for i in hard));examples=[l for p in parts for l in Path(str(p)+'.teacher.txt').read_text().splitlines()];assert examples;(root/'search-teacher.txt').write_text('\n'.join(examples)+'\n');print(json.dumps({'assessed':n,'hard':len(hard),'clearSpread':sum(r['clearSpread'] for r in rows),'teacherExamples':len(examples),'fourPlyTeachers':sum(r['teacherDepth']==4 for r in rows)}),flush=True)
