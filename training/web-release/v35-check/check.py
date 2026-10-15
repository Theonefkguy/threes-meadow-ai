"""Verify the game binary and the shipped weights against protocol.json (V35 web release check)."""
import hashlib, json
from pathlib import Path
P = json.loads(Path('training/web-release/v35-check/protocol.json').read_text())
for f, h in P['fileSHA256'].items():
    assert hashlib.sha256(Path(f).read_bytes()).hexdigest() == h, f'mismatch {f}'
print('ok')
