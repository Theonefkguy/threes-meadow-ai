import subprocess,concurrent.futures,json
from pathlib import Path
root=Path('training/v6');s=json.loads((root/'selection.json').read_text());arms=[('control',str(root/'run-control/checkpoint.ntd'),0)]
# V4 is evaluated independently alongside training; do not inspect until selection freezes.
if s['selected']!='control':arms.append(('selected',str(root/'selected.ntd'),int(s['kind']=='reach1536')))
def run(item):
 name,model,mode=item
 with open(root/f'holdout-{name}.log','w') as f:subprocess.run(['python','training/v6/run-eval.py',model,str(mode),'8410001','5000',str(root/f'holdout-{name}.jsonl'),str(8//len(arms))],stdout=f,stderr=subprocess.STDOUT,check=True)
 print(name+' finished',flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=len(arms)) as ex:list(ex.map(run,arms))
