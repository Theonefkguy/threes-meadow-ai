# Web release of five-cell residual models (preparation)

Tools for shipping a model with five-cell residuals (V29–V31) to the website. The website
default is unchanged until a release round adopts a model.

1. **Quantize** each stage residual for the web (float16, power-of-two scale) and write the
   same dequantized weights in the native format:
   `python3 scripts/quantize-residual.py IN.res OUT.f16 OUT_NATIVE.res`
2. **Native strength check** of the quantized weights (OUT_NATIVE.res) against the float32
   weights, with the round's whole-game or conditional evaluator.
3. **Parity**: build `reference.cpp`
   (`c++ -std=c++17 -O3 training/web-release/reference.cpp -o /tmp/web-reference`), run it on
   the 320 V25 browser states, then
   `node training/web-release/parity.mjs BASE.bin REF.jsonl OUT.f16 ...`. Directions and depths
   must match exactly and root values to a relative 1e-12 (V30 stage-2 residual: 0
   mismatches, max relative difference 7.6e-16; V23 without residual: 6.5e-16).
4. **Browser timing** on an idle machine: serve the repository root and open
   `training/web-release/browser/?base=...&res=...`; P95 must stay below the 160 ms budget
   with depth 5 completed on ≥ 99% of measurements, on desktop and on a phone.
5. **Ship**: copy BASE.bin and the .f16 files to `dist/models/`, add the version to
   `MODEL_FILES` in `dist/rl-model.js`, switch the worker's default, bump the `?v=` cache keys,
   and run `node --test tests/*.test.mjs` and `python3 scripts/verify-release.py`.

Web residual format (`dist/rl-model.js` `decodeResidual`): 32-byte header
`{0x3548444e, 4 patterns, 2^20 entries, stage, dtype 1 = float16, scaleExp (int32), 0, 0}`
followed by 4 × 2^20 little-endian float16 values; weight = half × 2^scaleExp.
