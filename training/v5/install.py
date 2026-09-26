from pathlib import Path
import json,shutil,hashlib
v=Path('training/v5');s=json.loads((v/'selection.json').read_text());p=json.loads((v/'protocol.json').read_text());model=v/'selected.ntd';assert hashlib.sha256(model.read_bytes()).hexdigest()==s['hashes'][s['winner']]
shutil.copyfile(model,'dist/models/ntuple-v5.bin');config={'version':5,'kind':'corner-potential','coefficient':s['coefficient'],'selectedArm':s['winner']};Path('dist/models/v5-policy.json').write_text(json.dumps(config,indent=2)+'\n')
m={'version':5,'selection':s,'protocol':p,'bytes':model.stat().st_size,'weightSemantics':'shaped return U; restore U+coefficient*potential in search'};Path('dist/models/ntuple-v5.json').write_text(json.dumps(m,indent=2)+'\n');print(config)
