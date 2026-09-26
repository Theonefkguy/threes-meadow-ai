import time,subprocess,json,concurrent.futures
from pathlib import Path
v=Path('training/v3')
while not all((v/f'screen-{n}.jsonl').exists() for n in ['a','b','c']):time.sleep(2)
subprocess.run(['python',str(v/'freeze-selection.py')],check=True)
selection=json.loads((v/'selection.json').read_text());goal=str(v/'selected.goal') if selection['goalSha256'] else '-'
def run(args,name):
 with open(v/(name+'.log'),'w') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
 print(name+' complete',flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 jobs=[pool.submit(run,['python',str(v/'run-eval.py'),str(v/'selected.ntd'),goal,'3910001','5000',str(v/'holdout-v3.jsonl'),'8'],'holdout-v3'),pool.submit(run,['python',str(v/'run-late-eval.py'),str(v/'selected.ntd'),goal,str(v/'late-v3.jsonl')],'late-v3')]
 for job in jobs:job.result()
while not (v/'holdout-v2.jsonl').exists():time.sleep(2)
subprocess.run(['python',str(v/'summarize.py')],check=True)
