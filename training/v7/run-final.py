import subprocess
from pathlib import Path
root=Path('training/v7')
with open(root/'holdout-selected.log','w') as f:subprocess.run(['python','training/v7/run-eval.py',str(root/'selected.ntd'),'0','9410001','5000',str(root/'holdout-selected.jsonl'),'6'],stdout=f,stderr=subprocess.STDOUT,check=True)
print('selected finished',flush=True)
