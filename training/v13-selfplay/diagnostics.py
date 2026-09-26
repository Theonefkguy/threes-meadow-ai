# Posthoc implementation/behavior controls prompted by zero initial successes.
from pathlib import Path
import concurrent.futures,subprocess,json
p=Path('training/v13-selfplay')
def run(j):
 mode,start=j;subprocess.run(['/tmp/threes-v13-bench','base',mode,str(start),'16',str(p/f'diagnostic-{mode}.part{start}.jsonl')],check=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:list(ex.map(run,[(m,s) for s in range(11910001,11910513,16) for m in ['table','two']]))
for m in ['table','two']:
 rs=sorted([json.loads(l) for f in p.glob(f'diagnostic-{m}.part*.jsonl') for l in f.read_text().splitlines()],key=lambda r:r['seed']);assert [r['seed'] for r in rs]==list(range(11910001,11910513));(p/f'diagnostic-{m}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rs))
