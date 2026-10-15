# V35 web release check (bonus-cycle rule)

| Check | Result | Files |
|---|---|---|
| Native strength of the shipped float16 V31 weights under the cycle rule, cycle-aware search, 2048 games on the V31 release seeds | 1536 99.4%, 3072 95.8%, 6144 65.1% [63.0, 67.1], 12288 6.7% [5.7, 7.9], mean score 609,920; not significantly different from V35 arm A (float32, 4096 games): 6144 p=0.26, 12288 p=0.50 → sanity check passes | `protocol.json`, `run.sh`, `check.py`, `analyze.py`, `results.json`, `games/` |
| JS vs C++ parity, 320 V25 browser states, with and without the cycle | 0 mismatches in direction and depth both ways; max root relative difference 7.6e-16 / 4.3e-16 | `parity-timing.json`, `reference-aware.jsonl` (`training/v35-cycle/web-reference.cpp`) |
| Timing | desktop Chromium, cycle-aware: mean 6.5 ms, P95 11.5 ms, depth 5 on 960/960; node, interleaved: aware 1.7% fewer nodes than unaware | `parity-timing.json` |
| Combination replay (same 2048 games replayed, every game identical to the record; first move with these cards on the board together) | 6144+3072 47.8% (979), 6144+3072+1536 33.2% (680), 6144+3072+1536+768 23.2% (475); all 137 games that made 12288 passed through 6144+3072+1536+768 | `combo.sh`, `training/v35-cycle/combo.cpp`, `combo.json`, `combo/` |

The protocol was committed before any game of the check (`39731ca`).
