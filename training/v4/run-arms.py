import time,subprocess,concurrent.futures
from pathlib import Path
v=Path('training/v4')
while not (v/'fresh-pool.txt').exists():time.sleep(2)
def run(name,args):
 with open(v/(name+'.log'),'w') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
 print(name+' complete',flush=True)
def arm(name,seed):
 run('train-'+name,['/tmp/threes-v4-train',str(v/('run-'+name)),name,'450',str(seed),'dist/models/ntuple-v3.bin',str(v/'fresh-pool.txt')])
 run('screen-'+name,['python',str(v/'run-eval.py'),str(v/f'run-{name}/checkpoint.ntd'),str(v/f'run-{name}/checkpoint.five') if name=='b' else '-','6310001','400',str(v/f'screen-{name}.jsonl'),'1'])
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
 futures=[ex.submit(arm,name,seed) for name,seed in [('a',202610012),('b',202610013),('c',202610014)]]
 for f in futures:f.result()
