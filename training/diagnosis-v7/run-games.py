from pathlib import Path
import subprocess, concurrent.futures, json
root=Path(__file__).resolve().parents[2]
p=root/'training/diagnosis-v7'
protocol=json.loads((p/'protocol.json').read_text())
def run(job):
 mode,start,n=job
 out=p/f'{mode}.part{start}.jsonl'
 subprocess.run(['/tmp/threes-diagnosis-games',str(root/'training/v7/base.ntd'),mode,str(start),str(n),str(out)],check=True)
 print(mode,start,n,flush=True)
 return out
jobs=[(mode,9510001+i,16) for i in range(0,512,16) for mode in protocol['policies']]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 for _ in pool.map(run,jobs):pass
for mode in protocol['policies']:
 rows=[json.loads(line) for f in sorted(p.glob(f'{mode}.part*.jsonl')) for line in f.read_text().splitlines()]
 rows.sort(key=lambda r:r['seed'])
 assert [r['seed'] for r in rows]==list(range(9510001,9510513))
 (p/f'{mode}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
