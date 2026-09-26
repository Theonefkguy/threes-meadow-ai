from pathlib import Path
import concurrent.futures,subprocess,json,hashlib
p=Path('training/v14-depth');pr=json.loads((p/'protocol.json').read_text());assert hashlib.sha256(Path(pr['base']).read_bytes()).hexdigest()==pr['baseSHA256']
def run(job):
 d,s=job;subprocess.run(['/tmp/threes-v14-bench',str(d),str(s),str(p/f'depth{d}.part{s}.jsonl'),str(p/f'depth{d}.part{s}.progress')],check=True);print('Completed',d,s,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:list(ex.map(run,[(d,s) for s in range(12010001,12010017) for d in [4,5]]))
for d in [4,5]:
 rows=sorted([json.loads(l) for f in p.glob(f'depth{d}.part*.jsonl') for l in f.read_text().splitlines()],key=lambda r:r['seed']);assert [r['seed'] for r in rows]==list(range(12010001,12010017));(p/f'depth{d}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
assert hashlib.sha256(Path(pr['base']).read_bytes()).hexdigest()==pr['baseSHA256'];print('Complete',flush=True)
