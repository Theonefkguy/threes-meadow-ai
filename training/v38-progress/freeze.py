"""Record pool and model hashes before evaluation (manifest.json); later runs must match it."""
import hashlib, json, time
from pathlib import Path
L = Path('training/v38-progress'); sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
files = [L / 'pool/first6144.txt'] + [L / f'models/{a}-final.{e}' for a in 'PCB' for e in ('bin', 'res', 'bias') if (L / f'models/{a}-final.{e}').exists()]
rec = {str(f): sha(f) for f in files}; man = L / 'manifest.json'
if man.exists():
    assert json.loads(man.read_text())['sha256'] == rec, 'pool or models changed after freezing'
else:
    w = Path('training/v31-stages/models/W31.bin').read_bytes(); S = 8 * 65536 * 4
    for a in 'PC':
        b = (L / f'models/{a}-final.bin').read_bytes(); assert b[:16] == w[:16] and len(b) == len(w)
        for s in (0, 1, 2): assert b[16 + s * S:16 + (s + 1) * S] == w[16 + s * S:16 + (s + 1) * S], f'{a} changed stage {s}'
    assert (L / 'models/B-final.bin').read_bytes() == w and sha(L / 'models/B-final.res') == sha('training/v31-stages/models/S3-final.res'), 'B changed tables'
    man.write_text(json.dumps({'frozenUTC': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'sha256': rec,
                               'note': 'written after pool generation and training, before any evaluation game; P/C stages 0-2 and all of B tables verified identical to W31'}, indent=2) + '\n')
print('frozen')
