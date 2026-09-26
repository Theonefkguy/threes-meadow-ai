from pathlib import Path
import json,hashlib
v=Path('training/v7');s=json.loads((v/'selection.json').read_text());b=(v/'selected.ntd').read_bytes();assert hashlib.sha256(b).hexdigest()==s['sha256'];Path('dist/models/ntuple-v7.bin').write_bytes(b)
