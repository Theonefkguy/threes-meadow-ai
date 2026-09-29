"""Run frozen V25 paired evaluation, checkpointed in 8-game chunks."""
from pathlib import Path
import concurrent.futures as cf
import subprocess,sys,os,time,json,hashlib
ROOT=Path(__file__).resolve().parent
p=json.loads((ROOT/'protocol.json').read_text());binary=Path(sys.argv[1]) if len(sys.argv)>1 else Path(p['binaryLocalPath'])
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(ROOT/'model.bin')==p['modelSHA256']
assert sha(binary)==p['binarySHA256'],'Compile and explicitly record a new binary for cross-platform reproduction'
for f,h in p['sourceSHA256'].items():assert sha(ROOT/f)==h
chunk=p['chunkSize'];first,last=p['seeds'];env=dict(os.environ);env.pop('THREES_BONUS_RULE',None)
start=time.time();jobs=[(arm,s) for s in range(first,last+1,chunk) for arm in ('baseline','adaptive')]
def run(job):
 arm,seed=job;output=ROOT/'data'/f'{arm}-{seed}.jsonl';n=min(chunk,last-seed+1)
 if output.exists():
  rows=[json.loads(l) for l in output.read_text().splitlines()]
  assert [r['seed'] for r in rows]==list(range(seed,seed+n))
  return
 temp=output.with_suffix('.part');temp.unlink(missing_ok=True)
 subprocess.run([str(binary),str(ROOT/'model.bin'),'5','0.01' if arm=='baseline' else '0.005','0','1',str(seed),str(n),str(temp)],check=True,env=env)
 rows=[json.loads(l) for l in temp.read_text().splitlines()];assert [r['seed'] for r in rows]==list(range(seed,seed+n))
 temp.rename(output)
def progress(status):
 completed=sum(len(f.read_text().splitlines()) for f in (ROOT/'data').glob('*.jsonl'))
 partial=sum(len(f.read_text().splitlines()) for f in (ROOT/'data').glob('*.part'))
 elapsed=time.time()-start
 obj={'status':status,'completedGames':completed+partial,'checkpointedGames':completed,'totalGames':2*p['sampleSizePerArm'],'elapsedSeconds':elapsed,'etaSeconds':max(0,(2*p['sampleSizePerArm']-completed-partial)*elapsed/(completed+partial)) if completed+partial else None}
 (ROOT/'progress.json').write_text(json.dumps(obj,indent=2));print(json.dumps(obj),flush=True)
try:
 with cf.ThreadPoolExecutor(max_workers=p['workers']) as pool:
  futures={pool.submit(run,j) for j in jobs}
  while futures:
   done,futures=cf.wait(futures,timeout=30,return_when=cf.FIRST_EXCEPTION)
   for f in done:f.result()
   progress('running' if futures else 'complete')
 for arm in ('baseline','adaptive'):
  rows=sorted([json.loads(l) for f in (ROOT/'data').glob(f'{arm}-*.jsonl') for l in f.read_text().splitlines()],key=lambda r:r['seed'])
  assert [r['seed'] for r in rows]==list(range(first,last+1))
  (ROOT/f'games-{arm}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 progress('complete')
except BaseException:
 progress('failed');raise
