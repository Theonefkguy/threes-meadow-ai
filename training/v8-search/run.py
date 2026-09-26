from pathlib import Path
import subprocess,concurrent.futures,json,sys,hashlib,datetime
p=Path('training/v8-search');protocol=json.loads((p/'protocol.json').read_text());assert hashlib.sha256(Path(protocol['model']).read_bytes()).hexdigest()==protocol['modelSHA256']
phase=sys.argv[1]; modes=protocol['arms'] if phase=='screen' else ['full',json.loads((p/'selection.json').read_text())['selected']];first,n=(9810001,512) if phase=='screen' else (9910001,1024)
def run(job):
 mode,start=job;out=p/f'{phase}-{mode}.part{start}.jsonl'
 subprocess.run(['/tmp/threes-v8-bench',protocol['model'],mode,str(start),'16',str(protocol['budgetCPUmsPerMove']),str(out)],check=True)
 print(mode,start,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
 for _ in ex.map(run,[(m,first+i) for i in range(0,n,16) for m in modes]):pass
for mode in modes:
 rows=sorted([json.loads(l) for f in p.glob(f'{phase}-{mode}.part*.jsonl') for l in f.read_text().splitlines()],key=lambda r:r['seed']);assert [r['seed'] for r in rows]==list(range(first,first+n));(p/f'{phase}-{mode}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
if phase=='screen':
 wins={m:sum(json.loads(l)['success'] for l in (p/f'screen-{m}.jsonl').read_text().splitlines()) for m in modes};selected=max(['beam2','adaptive','mcts'],key=lambda m:wins[m]);(p/'selection.json').write_text(json.dumps({'selected':selected,'screenWins':wins,'rule':protocol['selection'],'frozenAt':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2));print(wins,selected)
