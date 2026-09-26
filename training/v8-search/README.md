# Model-guided selective search experiment

Frozen V7 starting model; no new trained weights. The current website remains unchanged unless later validation supports promotion.

## Algorithms

- `full`: existing expectimax algorithm adapted to a1ms thread-CPU budget, iterative depths1–8, atomic fallback to last completed iteration. This is a time-matched reference, not exactly the deployed3ply/24000-node configuration.
- `beam2`: every root move searched; below root, all legal actions get a full two-ply score estimate before deeper search is limited to the top two. This is approximate forward pruning, not certified dominance pruning. All chance outcomes are still fully enumerated at their correct probabilities.
- `adaptive`: additionally deepen any action within1500 score units of the shallow best.
- `mcts`: model-derived softmax action priors, PUCT selection, each root action explored, true stochastic transitions sampled from public previews/counts; independent search RNG; transposition keys include remaining-depth context. New nodes bootstrap from the frozen score model. Most-visited root action returned. No hidden real deck information used.

The existing model estimates future score. Its board-only input and possible estimation errors are unchanged; the public search state includes previews/counts. This round isolates search changes and does not claim to have trained a new policy/value network or completed a search-learning feedback cycle.

## Protocol and fairness

See `protocol.json`, fixed before screening. Each candidate and the full reference receives the same1ms thread-CPU budget per move, including per-search cache cleanup/setup. Timers are checked periodically, so record actual usage and overruns. The existing first-depth fallback and MCTS root exploration minimum can slightly overrun the soft budget. An independent8-game-per-arm pilot checked timing only; pilot games are excluded from comparisons.

Screen512 common game seeds per arm. Select the best candidate once, then freeze it before1024 new paired test games against the full reference. All games stop at first1536 or actual death; truncation is an error. Depth distributions are descriptive: completed full-tree depth and the deepest sampled MCTS path are not equivalent quantities.

Native CPU-time budgets are hardware/load dependent, and these algorithms have not yet been ported or timed in a browser. Success at1536 alone does not establish6144 performance. Only a positive paired95% lower bound and acceptable time matching warrant subsequent late-game/browser validation. No secondary candidate is chosen after inspecting holdout results.

## Reproduction

From the repository root, restore `training/v7/base.ntd` from `training/v6/candidate-control.ntd.gz` if needed; checksum checked by runner.

```
g++ -std=c++17 -O3 -march=native training/v8-search/verify.cpp -o /tmp/threes-v8-verify
/tmp/threes-v8-verify
g++ -std=c++17 -O3 -march=native training/v8-search/bench.cpp -o /tmp/threes-v8-bench
python training/v8-search/run.py screen
python training/v8-search/run.py holdout
python training/v8-search/report.py
```

Verification checks exact-search reduction when selective omission is disabled, legal actions and root exploration, and empirical ordinary/bonus hint sampling versus its exact probabilities. Timed reruns can differ in actions; saved JSONL files preserve the measured trials.

## Follow-up and experimental website option

After the positive primary test, `late-protocol.json` fixed a secondary512-game cohort to6144, plus a descriptive current-V4 reference. Compile/run `late.cpp` via `run-late.py`, then run `report-late.py`, `report.py` and `write-report.py`. The secondary cohort did not replicate the early improvement. The pooled early estimate includeszero, and6144 has no demonstrated gain. V4 remains the website default.

The experimental `rl-v8` option uses the same frozen weights and a JavaScript port of PUCT, capped at640 simulations/160ms wall time. At1536 it switches to the established3ply/24000-node search. This browser budget differs from the native1msCPU research budget; offline win rates are not browser win rates. `parity.cpp` and `verify-js.mjs` verify160 fixed-simulation native/JS cases,8 late fallback cases and80 timing positions. `verify-worker.mjs` verifies45 worker loading/fallback cases. Timing uses Node, not an actual mobile browser. Managed static previews are unavailable; no live-Site browser QA was performed.

The native soft-budget runs contain rare large overshoots (candidate peak~198ms, full reference~88ms) despite closely matched averages. The budget is checked between search operations, not a hard real-time guarantee; actual per-game timing summaries are preserved. The browser runs search in a terminable Worker.
