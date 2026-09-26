from pathlib import Path
import subprocess, concurrent.futures, json, hashlib
p=Path('training/v9'); protocol=json.loads((p/'protocol.json').read_text()); manifest=json.loads((p/'manifest.json').read_text())
assert hashlib.sha256(Path(protocol['baseModel']).read_bytes()).hexdigest()==protocol['baseSHA256']
for name,meta in manifest['models'].items():
 assert hashlib.sha256((p/name/'best.net').read_bytes()).hexdigest()==meta['sha256']
names=['base']+list(manifest['models']); first,last=protocol['evaluationSeeds']
def run(job):
 name,start=job; out=p/f'eval-{name}.part{start}.jsonl'
 subprocess.run(['/tmp/threes-v9-bench','base' if name=='base' else str(p/name/'best.net'),name,str(start),'16','1',str(out)],check=True)
 print(name,start,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
 for _ in ex.map(run,[(name,start) for start in range(first,last+1,16) for name in names]): pass
for name in names:
 rows=sorted([json.loads(l) for f in p.glob(f'eval-{name}.part*.jsonl') for l in f.read_text().splitlines()],key=lambda r:r['seed'])
 assert [r['seed'] for r in rows]==list(range(first,last+1))
 (p/f'eval-{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
