import subprocess,json,hashlib
import numpy as np
from pathlib import Path
v=Path('training/v7');out={};base=(v/'base.ntd').read_bytes()
def run(name,args):
 p=subprocess.run(args,text=True,capture_output=True,check=True);out[name]=p.stdout.strip();print(name+': '+p.stdout.strip(),flush=True)
run('native',['/tmp/threes-v7-verify'])
for arm in ['r0','r10','r25','r10teacher']:
 b=(v/f'run-{arm}/checkpoint.ntd').read_bytes();assert b[16+8*65536*4:]==base[16+8*65536*4:];assert np.isfinite(np.frombuffer(b,dtype='<f4',offset=16)).all()
out['stageIsolation']='All four candidates retain byte-identical later stage tables.'
with open(v/'parity-selected.log','w') as f:subprocess.run(['/tmp/threes-v7-benchmark',str(v/'selected.ntd'),'0','9610001','3',str(v/'parity-selected.jsonl'),'1'],stdout=f,stderr=subprocess.STDOUT,check=True)
run('parity',['node','training/v7/verify-parity.mjs',str(v/'selected.ntd'),str(v/'parity-selected.jsonl'),'0'])
run('worker',['node','training/v7/verify-worker.mjs']);run('input',['node','--test','tests/input.test.mjs']);run('timing',['node','training/v7/verify-timing.mjs']);(v/'validation.json').write_text(json.dumps(out,indent=2)+'\n')
