"""Verify sources, binaries and base model files against protocol.json (V38)."""
import hashlib, json
from pathlib import Path
P = json.loads(Path('training/v38-progress/protocol.json').read_text())
for f, h in P['fileSHA256'].items(): assert hashlib.sha256(Path(f).read_bytes()).hexdigest() == h, f'mismatch {f}'
print('ok')
