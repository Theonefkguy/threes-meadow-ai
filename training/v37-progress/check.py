"""Verify sources, binary, base model and the V33 records against protocol.json (V37)."""
import hashlib, json
from pathlib import Path
L = Path('training/v37-progress'); P = json.loads((L / 'protocol.json').read_text())
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
for f, h in P['fileSHA256'].items(): assert sha(f) == h, f'mismatch {f}'
h = hashlib.sha256()
for f in sorted(Path('training/v33-patterns/data').glob('*.bin')): h.update(f.name.encode()); h.update(f.read_bytes())
assert h.hexdigest() == P['data']['combinedSHA256'], 'V33 records changed'
print('ok')
