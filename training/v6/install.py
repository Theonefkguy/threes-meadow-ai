from pathlib import Path
import json,hashlib
root=Path('training/v6');s=json.loads((root/'selection.json').read_text());blob=(root/'selected.ntd').read_bytes();assert hashlib.sha256(blob).hexdigest()==s['sha256'];Path('dist/models/ntuple-v6.bin').write_bytes(blob);Path('dist/models/v6-policy.json').write_text(json.dumps({'kind':s['kind'],'selectedArm':s['selected'],'target':1536,'handoff':'V4 at1536'},indent=2)+'\n')
