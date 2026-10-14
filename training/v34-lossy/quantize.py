#!/usr/bin/env python3
"""V34 lossy model encodings (standard library only).

python3 training/v34-lossy/quantize.py base16  IN.bin OUT_WEB OUT_NATIVE.bin
    four-cell tables -> float16 with a power-of-two scale per stage. OUT_WEB: 32-byte header
    {0x3148344e, stages, patterns, table, dtype=1, 0, 0, 0} + int8 scaleExp per stage (padded to
    16 bytes) + float16 values. OUT_NATIVE: the same dequantized weights in the native format.
python3 training/v34-lossy/quantize.py palette|palette-log IN.res OUT_WEB OUT_NATIVE.res
    five-cell residual (float32, the shipped float16 values) -> 8-bit codes into a 256-entry
    float32 codebook: code 0 is exactly 0 (unvisited entries), codes 1-255 come from 1-D k-means
    (Lloyd, quantile start, 30 iterations) on the nonzero values. OUT_WEB: 32-byte header
    {0x3850354e, patterns, table, stage, dtype=2, 0, 0, 0} + codebook (256 float32) + uint8 codes.
    OUT_NATIVE: dequantized weights in the native residual format.
    palette-log: the same, but the k-means runs on sign(v)*log1p(|v|) (relative precision for
    large and small weights); each code's value is the mean of its original weights.
"""
import math
import array, bisect, gzip, itertools, struct, sys

def base16(src, web, native):
    raw = open(src, 'rb').read(); magic, stages, patterns, table = struct.unpack('<4I', raw[:16])
    assert magic == 0x3144544e and len(raw) == 16 + stages * patterns * table * 4
    w = array.array('f'); w.frombytes(raw[16:]); n = patterns * table; out_half = b''; deq = array.array('f'); exps = []
    for s in range(stages):
        part = w[s * n:(s + 1) * n]; peak = max(abs(v) for v in part); e = 0
        while peak / 2 ** e > 65000: e += 1
        h = struct.pack(f'<{n}e', *(v / 2 ** e for v in part)); out_half += h; exps.append(e)
        deq.extend(v * 2 ** e for v in struct.unpack(f'<{n}e', h))
    with open(web, 'wb') as f:
        f.write(struct.pack('<8I', 0x3148344e, stages, patterns, table, 1, 0, 0, 0)); f.write(struct.pack('<16b', *(exps + [0] * (16 - len(exps))))); f.write(out_half)
    with open(native, 'wb') as f:
        f.write(raw[:16]); f.write(deq.tobytes())
    err = max(abs(a - b) for a, b in zip(w, deq)); print(f'base16: scaleExp {exps}, max abs error {err:.3f}')

def palette(src, web, native, log=False):
    raw = open(src, 'rb').read(); magic, patterns, table, stage = struct.unpack('<4I', raw[:16])
    assert magic == 0x3554444e and len(raw) == 16 + patterns * table * 4
    w = array.array('f'); w.frombytes(raw[16:])
    f = (lambda v: math.copysign(math.log1p(abs(v)), v)) if log else (lambda v: v)
    pairs = sorted((f(v), v) for v in w if v != 0.0); nz = [p[0] for p in pairs]; orig = [p[1] for p in pairs]; m = len(nz); K = 255
    prefix = list(itertools.accumulate(nz, initial=0.0))
    cent = [nz[min(m - 1, int((i + 0.5) * m / K))] for i in range(K)]
    for _ in range(30):
        cent = sorted(set(cent)); bounds = [(a + b) / 2 for a, b in zip(cent, cent[1:])]
        cuts = [0] + [bisect.bisect_right(nz, b) for b in bounds] + [m]
        cent = [(prefix[hi] - prefix[lo]) / (hi - lo) if hi > lo else c for c, lo, hi in zip(cent, cuts, cuts[1:])]
    cent = sorted(set(cent)); bounds = [(a + b) / 2 for a, b in zip(cent, cent[1:])]
    cuts = [0] + [bisect.bisect_right(nz, b) for b in bounds] + [m]
    po = list(itertools.accumulate(orig, initial=0.0))  # code value = mean of original weights in the cell
    vals = list(array.array('f', [(po[hi] - po[lo]) / (hi - lo) for lo, hi in zip(cuts, cuts[1:]) if hi > lo]))
    keep = [i for i, (lo, hi) in enumerate(zip(cuts, cuts[1:])) if hi > lo]; bounds = [bounds[i] for i in keep[:-1]]
    assert len(vals) <= K and len(bounds) == len(vals) - 1
    book = [0.0] + vals + [vals[-1]] * (K - len(vals))
    codes = bytearray(len(w))
    for i, v in enumerate(w):
        if v != 0.0: codes[i] = 1 + bisect.bisect_right(bounds, f(v))
    deq = array.array('f', (book[c] for c in codes))
    with open(web, 'wb') as f:
        f.write(struct.pack('<8I', 0x3850354e, patterns, table, stage, 2, 0, 0, 0)); f.write(array.array('f', book).tobytes()); f.write(codes)
    with open(native, 'wb') as f:
        f.write(raw[:16]); f.write(deq.tobytes())
    rel = sum((a - b) ** 2 for a, b in zip(w, deq)) / max(1e-12, sum(a * a for a in w))
    print(f'palette{"-log" if log else ""} stage {stage}: {len(vals)} levels, relative squared error {rel:.2e}, max abs error {max(abs(a - b) for a, b in zip(w, deq)):.1f}')

if __name__ == '__main__':
    if len(sys.argv) != 5 or sys.argv[1] not in ('base16', 'palette', 'palette-log'):
        raise SystemExit(__doc__)
    if sys.argv[1] == 'palette-log': palette(*sys.argv[2:], log=True)
    else: {'base16': base16, 'palette': palette}[sys.argv[1]](*sys.argv[2:])
