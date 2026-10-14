"""Verify the arm's model files against protocol.json before running (V34)."""
import hashlib, json, sys
from pathlib import Path
L = Path('training/v34-lossy'); P = json.loads((L / 'protocol.json').read_text())
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
for f in P['arms'][sys.argv[1]]['files']:
    assert sha(f) == P['fileSHA256'][f], f'hash mismatch {f}'
assert sha('training/v31-stages/bin/game') == P['gameBinarySHA256']
print('ok', sys.argv[1])
