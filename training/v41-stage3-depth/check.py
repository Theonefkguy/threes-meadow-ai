"""Verify files against protocol.json (V41)."""
import hashlib, json
from pathlib import Path
P = json.loads(Path('training/v41-stage3-depth/protocol.json').read_text())
for f, h in P['fileSHA256'].items(): assert hashlib.sha256(Path(f).read_bytes()).hexdigest() == h, f'mismatch {f}'
print('ok')
