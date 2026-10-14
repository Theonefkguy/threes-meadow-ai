#!/usr/bin/env python3
"""Losslessly gzip model files for the website (deterministic: no name, mtime 0, level 9).

python3 scripts/compress-models.py FILE [FILE ...]   -> writes FILE.gz next to each file

Prints each file's raw and compressed size and the SHA-256 of both. The browser decompresses
with DecompressionStream('gzip'), so the decoded bytes are identical to FILE.
"""
import gzip, hashlib, sys
from pathlib import Path

for name in sys.argv[1:]:
    src = Path(name); raw = src.read_bytes()
    out = Path(name + '.gz')
    with open(out, 'wb') as f, gzip.GzipFile(filename='', mode='wb', compresslevel=9, fileobj=f, mtime=0) as g:
        g.write(raw)
    assert gzip.decompress(out.read_bytes()) == raw
    print(f'{src.name}: {len(raw)} -> {out.stat().st_size} bytes; raw {hashlib.sha256(raw).hexdigest()[:16]}, gz {hashlib.sha256(out.read_bytes()).hexdigest()[:16]}')
