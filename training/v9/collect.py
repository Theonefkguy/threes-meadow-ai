from pathlib import Path
import subprocess,concurrent.futures,json
p=Path('training/v9')
def run(i):
 prefix=p/f'collection.part{i}';subprocess.run(['/tmp/threes-v9-collect',str(11010001+i*50),'50',str(prefix)],check=True);print(i,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:list(ex.map(run,range(24)))
(p/'teacher.txt').write_text(''.join((p/f'collection.part{i}.txt').read_text() for i in range(24)))
(p/'collection-games.jsonl').write_text(''.join((p/f'collection.part{i}.games.jsonl').read_text() for i in range(24)))
rows=[json.loads(l) for l in (p/'collection-games.jsonl').read_text().splitlines()];assert len(rows)==1200
print({'games':len(rows),'samples':sum(r['samples'] for r in rows),'wins':sum(r['success1536'] for r in rows)})
