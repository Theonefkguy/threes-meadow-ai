from pathlib import Path
import subprocess,time,json,concurrent.futures
v=Path('training/v5');p=json.loads((v/'protocol.json').read_text())
def run(name,label):
 with (v/f'holdout-{label}.log').open('w') as f:subprocess.run(['python',str(v/'run-eval.py'),str(v/f'run-{name}/checkpoint.ntd'),str(p['coefficients'][name]),'7410001','5000',str(v/f'holdout-{label}.jsonl'),'6'],stdout=f,stderr=subprocess.STDOUT,check=True)
 print('holdout '+label+' complete',flush=True)
# The control is pre-specified, so compute its holdout concurrently with screening.
# Do not inspect outcomes until the screen-only selection is frozen.
while not (v/'candidate-control.ntd.gz').exists():time.sleep(2)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
 control=ex.submit(run,'control','control')
 while not all((v/f'screen-{n}.jsonl').exists() for n in p['coefficients']):time.sleep(2)
 subprocess.run(['python',str(v/'freeze.py')],check=True);s=json.loads((v/'selection.json').read_text())
 shaping=ex.submit(run,s['bestShaping'],'shaping')
 control.result();shaping.result()
while not (v/'holdout-v4.jsonl').exists():time.sleep(2)
subprocess.run(['python',str(v/'summarize.py')],check=True)
