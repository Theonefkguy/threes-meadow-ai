import json
from pathlib import Path
v=Path('training/v3');e=json.loads((v/'evaluation.json').read_text());p=Path('dist/index.html');s=p.read_text()
if e['promote']:s=s.replace('<option value="rl-v3">','<option value="rl-v3" selected>').replace('<option value="rl-v2" selected>','<option value="rl-v2">')
else:s=s.replace('强化学习 · 第三版</option>','强化学习 · 第三版（实验）</option>')
p.write_text(s)
p=Path('dist/models/ntuple-v3.json');m=json.loads(p.read_text());m['holdout']=e;m['limitations']='Offline normal-opening tests have no wall-clock cutoff; browser uses a 160 ms soft budget. Reconstructed Threes bonus hint posterior is not verified against official app.';p.write_text(json.dumps(m,indent=2)+'\n')
print('Default V3:',e['promote'])
