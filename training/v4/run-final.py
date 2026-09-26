from pathlib import Path
import subprocess,time,json
v=Path('training/v4')
while not all((v/f'screen-{n}.jsonl').exists() for n in ['v3','control','a','b','c']):time.sleep(2)
subprocess.run(['python',str(v/'freeze.py')],check=True);s=json.loads((v/'selection.json').read_text());extra=str(v/'selected.five') if s['residualSha256'] else '-'
with open(v/'holdout-v4.log','w') as f:subprocess.run(['python',str(v/'run-eval.py'),str(v/'selected.ntd'),extra,'6410001','5000',str(v/'holdout-v4.jsonl'),'8'],stdout=f,stderr=subprocess.STDOUT,check=True)
while not (v/'holdout-v3.jsonl').exists():time.sleep(2)
subprocess.run(['python',str(v/'summarize.py')],check=True)
