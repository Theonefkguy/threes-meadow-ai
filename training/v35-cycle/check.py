"""Verify sources, binaries and model files against protocol.json (V35)."""
import hashlib, json
from pathlib import Path
L = Path('training/v35-cycle'); P = json.loads((L / 'protocol.json').read_text())
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
for group in ('sourceSHA256', 'binarySHA256', 'modelSHA256'):
    for f, h in P[group].items(): assert sha(f) == h, f'{group} mismatch {f}'
print('ok')
