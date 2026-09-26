# V3: 6144 curriculum experiments

Goal: improve normal-opening P(reach 6144), holding the reconstructed engine and public-information expectimax budget constant (3 plies / 24000 nodes; no time cutoff offline).

## Preregistered protocol

See protocol.json. Screening seeds 2910001–2910200; independent final evaluation 3910001–3915000. Select from completed A/B/C on screening only and freeze all weight hashes before opening holdout results. Retain V2 as default if independent evidence does not support improvement. Tests stop at 6144 or actual terminal; a 6000-move cap is separately reported. Training uses historical training trajectories, never evaluation trajectories.

A: 200000 additional episodes from 5000 real 3072 snapshots, five-step TD(lambda=.5), score objective, only late stage updated. Starts stratified on empty count and coexisting 1536; 20% recent on-policy replay after available. Episodes train through real terminal (including play after 6144), preserving the existing full-score value semantics. Earlier stages frozen.

B: frozen A score model plus a separately learned sigmoid probability head, target first 6144. 100000 late episodes, 2% random legal exploration, shuffled hidden remaining deck at restart; hidden order is never an input to the policy. Five-step TD with absorbing success 1 / failure 0. Auxiliary weight 200000, gated to late stage. This head estimates success under its training policy, not a guaranteed calibrated probability for 3-ply play. Probability-only learning is not used. Earlier values remain A's score objective.

C: 100000 additional score episodes from A, half normal and half real 1536 starts, using updated A late value as the 3072 bootstrap. Earlier stage weights updated; A late stage frozen. Same multi-step algorithm as V2.

These are sequential candidate experiments with differing episode counts, not a compute-matched causal ablation. A win does not prove any one ingredient is responsible.

## Evaluator acceleration

`ntuple-fast.h` is a copy of the unchanged engine with precomputed score powers. `search.h` aliases depth-1 cache keys by board because all leaf utilities here depend only on board. Every decision still ticks once; deeper keys retain hint posterior and bag counts. Verified exact full-game results, moves, node counts, completed depth counts against the original implementation before use. No fast-math compilation.

## Limits

Reconstructed original-game bonus preview posterior is not confirmed identical to the official app. Offline results exclude the browser's 160 ms soft cutoff. Historical curriculum state coverage differs from strong 3-ply runtime play. Any post-selection performance analysis must not tune against final holdout seeds.

## Reproduction

Run from the repository root. Requires C++17, Python 3, Node.js; SciPy is optional for the exact paired p-value. Build tools with `python training/v3/build.py`. Restore the 1536 training pool with `python training/v2/prepare-curriculum.py`. The actual 3072 training pool is saved as `training/v3/late-pool.txt`.

For evaluation without retraining, decompress the archived candidate checkpoints, use `dist/models/ntuple-v2.bin` as baseline, then run `python training/v3/run-eval.py MODEL GOAL_OR_DASH FIRST_SEED GAMES OUTPUT_JSONL WORKERS`. `run-eval.py` writes parts during execution and writes the combined result only after all workers finish successfully. To evaluate the selected score-only V3, use `dist/models/ntuple-v3.bin` and `-` for the goal argument.

Training entry points:

- `/tmp/threes-v3-train-late OUTPUT 200000 202609301 dist/models/ntuple-v2.bin training/v3/late-pool.txt`
- `/tmp/threes-v3-train-goal OUTPUT 100000 202609302 A_CHECKPOINT training/v3/late-pool.txt`
- `/tmp/threes-v3-train-joint OUTPUT 100000 202609303 A_CHECKPOINT training/v2/initial-curriculum.txt`

The fixed late test set is persisted in `late-eval.txt`, with collection provenance in its JSON sidecar. `eval-late.cpp` resamples latent remaining order per rollout; the policy does not read it. `verify-parity.mjs` compares native and actual browser policy directions, depths and node counts on sampled probe positions.

To regenerate the shipped-policy browser parity probes:

```
/tmp/threes-v3-benchmark-fast dist/models/ntuple-v3.bin - 2910001 3 training/v3/parity-selected.jsonl 1
node training/v3/verify-parity.mjs dist/models/ntuple-v3.bin training/v3/parity-selected.jsonl
node training/v3/verify-worker.mjs
```

The B probability head was screened offline and is not part of the selected browser build. The deployed learned-search algorithm is unchanged from V2 apart from module cache versions; the improvements in the main comparison come from learned weights, not deeper search or a larger node budget.
