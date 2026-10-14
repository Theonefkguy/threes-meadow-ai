# V31 web release checks

| Check | Result | Files |
|---|---|---|
| 1. Native strength of the float16 weights vs float32 (2048 V31 games) | 1884 games identical; 6144 30 vs 21 (p=0.26), 12288 10 vs 16 (p=0.33), score −1,359 [−7,409, +4,691]: passes | `protocol.json`, `run.sh`, `one.sh`, `analyze.py`, `results.json` |
| 2. JS vs C++ parity, 320 V25 browser states (64 at 1536, 128 at 3072, 128 at 6144) | 0 mismatches in direction and depth; max root relative difference 7.6e-16 | `reference-v31q.jsonl`, `parity.json` |
| 3. Browser timing, desktop Chromium 152 on the M5 Pro Mac mini | V31 mean 5.4 ms, P95 9.0 ms, max 11.9 ms, depth 5 on 960/960 (V23: 4.1 / 6.8 / 9.0 ms); phone not measured | `browser-desktop.json` |
| 4. Previous website AI (V23) vs shipped V31, same 2048 openings | 6144 38.6% → 59.3% (p=1e-44), 12288 3.9% → 5.6% (p=0.009), score 438,894 → 561,463 (p=3e-33), all Holm-significant | `release-protocol.json`, `one-v23.sh`, `release-analyze.py`, `release-results.json` |

Each check's rule was committed before it ran. Check 1's runner was fixed twice before any game
completed (an `xargs` command-length limit, then macOS `seq` printing seeds in scientific
notation); the acceptance rule did not change. Shipped files: `dist/models/ntuple-v31.bin`
(SHA-256 `b94a7da2…`, = V31 W31.bin) and `ntuple-v31-s{1,2,3}.f16` from
`scripts/quantize-residual.py` on V31 S1-final.res, V30 P-final.res and V31 S3-final.res.
