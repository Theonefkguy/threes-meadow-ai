from pathlib import Path
import concurrent.futures,subprocess,json
p=Path('training/v8-search')
def run(job):
 m,seed=job;model='dist/models/ntuple-v4.bin' if m=='v4' else 'training/v7/base.ntd';out=p/f'late-{m}.part{seed}.jsonl';subprocess.run(['/tmp/threes-v8-late',model,m,str(seed),'16',str(out)],check=True);print(m,seed,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
 for _ in ex.map(run,[(m,10010001+i) for i in range(0,512,16) for m in ['full','mcts','v4']]):pass
for m in ['full','mcts','v4']:
 rows=sorted([json.loads(l) for f in p.glob(f'late-{m}.part*.jsonl') for l in f.read_text().splitlines()],key=lambda r:r['seed']);assert [r['seed'] for r in rows]==list(range(10010001,10010513));(p/f'late-{m}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
