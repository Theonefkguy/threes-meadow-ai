"""Verify sources, binary, pool and models against protocol.json (V40)."""
import hashlib, json
from pathlib import Path
P = json.loads(Path('training/v40-late-depth/protocol.json').read_text())
for f, h in P['fileSHA256'].items(): assert hashlib.sha256(Path(f).read_bytes()).hexdigest() == h, f'mismatch {f}'
print('ok')
