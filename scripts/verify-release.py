"""Validate frozen V24 evidence without changing the archived experiment files."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REL = Path('training/v24-acceptance')

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def equivalent(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=1e-12, abs_tol=0)
    return a == b

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, help='Compiled V24 evaluator; rerun one seed per arm')
    args = parser.parse_args()
    protocol = json.loads((ROOT / REL / 'protocol.json').read_text())
    for path, digest in protocol['modelSHA256'].items():
        require(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, f'Model hash mismatch: {path}')
    require((ROOT/'training/v7/base.ntd').read_bytes() == (ROOT/'dist/models/ntuple-v8.bin').read_bytes(), 'Training base mismatch')
    first, last = protocol['seeds']
    data = {}
    for arm in ('OLD', 'NEW'):
        rows = [json.loads(line) for line in (ROOT / REL / f'games-{arm}.jsonl').read_text().splitlines()]
        require(len(rows) == protocol['sampleSize'], f'Wrong row count: {arm}')
        require(sorted(r['seed'] for r in rows) == list(range(first, last + 1)), f'Missing or duplicate seeds: {arm}')
        data[arm] = {row['seed']: row for row in rows}
    require(sum(r['r6144'] for r in data['NEW'].values()) == 400, 'NEW 6144 count mismatch')
    require(sum(r['r12288'] for r in data['NEW'].values()) == 31, 'NEW 12288 count mismatch')
    with tempfile.TemporaryDirectory(prefix='threes-v24-') as tmp:
        tmp = Path(tmp)
        target = tmp / REL
        target.mkdir(parents=True)
        for name in ('protocol.json', 'analyze.py', 'games-OLD.jsonl', 'games-NEW.jsonl'):
            shutil.copy2(ROOT / REL / name, target / name)
        subprocess.run([sys.executable, str(target/'analyze.py')], cwd=tmp, check=True, stdout=subprocess.DEVNULL)
        require(equivalent(json.loads((ROOT/REL/'results.json').read_text()), json.loads((target/'results.json').read_text())), 'Re-analysis differs')
        if args.replay:
            binary = args.replay.resolve()
            configs = {'OLD': ['ntuple-v4.bin', '3', '0', '24000', '0'], 'NEW': ['ntuple-v23.bin', '5', '0.01', '0', '1']}
            for arm, (model, *config) in configs.items():
                output = tmp / f'replay-{arm}.jsonl'
                subprocess.run([str(binary), str(ROOT/'dist/models'/model), *config, str(first), '1', str(output)], check=True, cwd=ROOT)
                rows = output.read_text().splitlines()
                require(len(rows) == 1, f'Unexpected replay length: {arm}')
                actual, expected = json.loads(rows[0]), data[arm][first]
                for key in expected:
                    if key not in ('searchCPU', 'maxMoveCPU'):
                        require(actual[key] == expected[key], f'Replay differs: {arm}/{key}')
                print(f'{arm} seed {first}: every non-timing field reproduced')
    print('PASS: model hashes, seed coverage, milestone counts and archived statistics')

if __name__ == '__main__':
    main()
