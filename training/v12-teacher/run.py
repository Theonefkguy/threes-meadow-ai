from pathlib import Path
import json,hashlib,subprocess,concurrent.futures
p=Path('training/v12-teacher');pr=json.loads((p/'protocol.json').read_text());assert hashlib.sha256(Path(pr['base']).read_bytes()).hexdigest()==pr['baseSHA256']
def run(i):
 subprocess.run(['/tmp/threes-v12-experiment',str(11710001+8*i),'8',str(p/f'audit.part{i}')],check=True);print('Completed source games',8*i+1,'to',8*i+8,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:list(ex.map(run,range(32)))
for suffix in ['states.jsonl','rollouts.jsonl','snapshots.txt']:
 (p/suffix).write_text(''.join((p/f'audit.part{i}.{suffix}').read_text() for i in range(32)))
states=[json.loads(l) for l in (p/'states.jsonl').read_text().splitlines()];assert [s['seed'] for s in states]==list(range(11710001,11710257));assert len((p/'snapshots.txt').read_text().splitlines())==256
assert hashlib.sha256(Path(pr['base']).read_bytes()).hexdigest()==pr['baseSHA256'];print('Complete',flush=True)
