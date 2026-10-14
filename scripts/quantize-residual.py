#!/usr/bin/env python3
"""Quantize a native five-cell residual (training/v29-five/model5.h format, float32) for the web.

python3 scripts/quantize-residual.py IN.res OUT_WEB.f16 OUT_NATIVE.res

OUT_WEB: 32-byte header {0x3548444e, patterns, table, stage, dtype=1, scaleExp, 0, 0} and
float16 values (round to nearest even); weight = half * 2^scaleExp, with the smallest
scaleExp >= 0 that keeps every value finite. OUT_NATIVE: the same dequantized weights in
the native float32 format, so C++ evaluates exactly the weights the browser uses.
Standard library only.
"""
import array, struct, sys

MAGIC_NATIVE, MAGIC_WEB = 0x3554444e, 0x3548444e

def main(src, web, native):
    raw = open(src, 'rb').read()
    magic, patterns, table, stage = struct.unpack('<4I', raw[:16])
    assert magic == MAGIC_NATIVE and len(raw) == 16 + patterns * table * 4, 'not a native residual'
    w = array.array('f'); w.frombytes(raw[16:])
    peak = max(abs(v) for v in w)
    exp = 0
    while peak / 2 ** exp > 65000:  # margin below the float16 maximum 65504 for rounding
        exp += 1
    scaled = [v / 2 ** exp for v in w]  # exact: power-of-two scaling of float32 values
    half = struct.pack(f'<{len(scaled)}e', *scaled)
    back = struct.unpack(f'<{len(scaled)}e', half)
    deq = array.array('f', (v * 2 ** exp for v in back))
    assert all(v == v and abs(v) != float('inf') for v in deq)
    with open(web, 'wb') as f:
        f.write(struct.pack('<5Ii2I', MAGIC_WEB, patterns, table, stage, 1, exp, 0, 0)); f.write(half)
    with open(native, 'wb') as f:
        f.write(struct.pack('<4I', MAGIC_NATIVE, patterns, table, stage)); f.write(deq.tobytes())
    err = max(abs(a - b) for a, b in zip(w, deq))
    print(f'stage {stage}: scaleExp {exp}, peak {peak:.1f}, max abs rounding error {err:.4f}, web bytes {32 + len(half)}')

if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
