import subprocess,json
from pathlib import Path
root=Path('training/v6');selection=json.loads((root/'selection.json').read_text());results={}
def run(name,args):
 r=subprocess.run(args,text=True,capture_output=True,check=True);results[name]=r.stdout.strip();print(name+': '+r.stdout.strip(),flush=True)
run('native',['/tmp/threes-v6-verify'])
run('model',['node','training/v6/verify-model.mjs'])
for name,model,mode in [('selected',str(root/'selected.ntd'),int(selection['kind']=='reach1536')),('probability',str(root/'run-probability/checkpoint.ntd'),1)]:
 # This range is reserved for browser/native equivalence, not policy selection.
 with open(root/f'parity-{name}.log','w') as log:subprocess.run(['/tmp/threes-v6-benchmark',model,str(mode),'8610001','3',str(root/f'parity-{name}.jsonl'),'1'],stdout=log,stderr=subprocess.STDOUT,check=True)
 run('parity-'+name,['node','training/v6/verify-parity.mjs',model,str(root/f'parity-{name}.jsonl'),str(mode)])
run('worker',['node','training/v6/verify-worker.mjs'])
run('input',['node','--test','tests/input.test.mjs'])
run('timing',['node','training/v6/verify-timing.mjs'])
(root/'validation.json').write_text(json.dumps(results,indent=2)+'\n')
