import subprocess,concurrent.futures
from pathlib import Path
root=Path('training/v7')
def run(item):
 name,model=item
 with open(root/f'holdout-{name}.log','w') as f:subprocess.run(['python','training/v7/run-eval.py',model,'0','9410001','5000',str(root/f'holdout-{name}.jsonl'),'4'],stdout=f,stderr=subprocess.STDOUT,check=True)
 print(name+' complete (outcomes sealed until selection)',flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(run,[('baseline',str(root/'base.ntd')),('v4','dist/models/ntuple-v4.bin')]))
