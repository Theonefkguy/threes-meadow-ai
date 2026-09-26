# V4 controlled late-game learning experiments

Goal: improve the probability of reaching 6144 from normal openings, preserving the canonical reconstructed engine and runtime search (3 plies, 24000 nodes, browser 160 ms soft limit). The previous model is V3.

## Protocol

`protocol.json` locks data sources, training budgets, seeds, candidate selection and promotion. Four trainers get 450 single-process CPU seconds each (initial file loading excluded, final episode completed). This is compute-budget matched, not episode matched. The shared fresh-pool collection cost is additional and reported separately. CPU-based stopping can yield slightly different episode counts on reruns; archived final weights and logs define the evaluated models.

- **Control**: additional V3 late-stage training using the historical 5000 late positions and 2-ply training policy.
- **A**: fresh positions collected using V3 3-ply play; 5% entire training episodes use the same 3-ply / 24000-node policy, 95% use the cheap 2-ply policy. Five-step TD(lambda=.5), only the late table changes.
- **B**: A's data/policy mixture; freeze all V3 weights and learn a zero-initialized residual with two five-cell patterns `{0,1,2,3,4}` and `{0,1,2,4,5}`, eight D4 symmetries each. Each table has 16^5 Float32 entries. Adds 8 MiB to the 6 MiB base. Only late-stage evaluation uses it.
- **C**: A's data/policy mixture; update existing late weights using a Temporal Coherence multiplier `abs(sum signed errors)/sum absolute errors` per parameter (1 when unseen), with symmetry multiplicities preserved. Nominal learning rate is 2.5 times A's, before this multiplier. This combines TC with the existing five-step forward lambda target; it is not a reproduction of the paper's complete online eligibility algorithm.

All arms: start from V3, stratify starts by empty count and coexisting 1536, 20% on-policy recent-position replay, shuffle the remaining hidden deck on restart conditional on observed composition, train through actual terminal to preserve full-score value semantics. Hidden order is simulator-only; the policy sees board, public hint, and public remaining counts. Learning-rate schedule uses consumed CPU-budget fractions (60%,85%); A/B/control use .02/.008/.003.

## Data and tests

Fresh training collection: 1000 normal V3 games, seeds 6110001–6111000, snapshot at entry to 3072 and every 40 moves thereafter, up to eight positions per source game; stop collection at 6144 or terminal. `fresh-pool.sources.jsonl` retains source-game identity. No final-test trajectories enter training or selection.

Screen: 400 normal openings each, seeds 6310001–6310400. Select highest 6144 count; ties prefer control,A,C,B. Freeze hashes before reading the new 5000-seed holdout (6410001–6415000). Promotion requires a positive paired difference with 95% lower bound above zero and browser compatibility checks. A/B/C are bundled interventions: A changes data and training search, C also changes nominal learning rate, B changes capacity and trainable weights. Do not attribute a gain to a single isolated cause.

## Source

TC's signed/absolute-error adaptation follows the idea in Jaskowski, *Mastering 2048 with Delayed Temporal Coherence Learning, Multi-Stage Weight Promotion, Redundant Encoding and Carousel Shaping*, Algorithm 2: https://arxiv.org/pdf/1604.05085 . That study concerns 2048, not a guarantee of gains in Threes.

## Reproduction

Run from the project root. Compile C++17 with `-O3 -march=native` (no fast-math). `build.py` produces `/tmp/threes-v4-*`. Restore archived candidate `.ntd.gz` and `.five.gz` with gzip before evaluation. `run-eval.py MODEL RESIDUAL_OR_DASH FIRST_SEED GAMES OUTPUT WORKERS` runs the exact fixed-budget evaluator. Both the original four-cell weights and five-cell residual weights use explicit magic/size headers.

The engine is reused from V3. Offline C++ and browser JS must agree on directions, node counts and completed depth on sampled positions. Browser timing checks use the actual 160 ms budget; they are not a phone-specific success-rate measurement. Reconstructed bonus-preview rules have not been independently proven identical to the official Threes app.

The browser now shares depth-1 cache entries across hint/count combinations because the leaf decision depends only on the board. This matches the native evaluator's existing optimization. It applies equally to V1/V2/V3/V4 score policies, leaves tick/node counting unchanged, and is validated against native directions/depths/nodes. It reduces redundant computation; offline strength comparisons still isolate model changes under the same node budget. The separate context-aware experimental V2 goal search is unaffected.

### Re-evaluate the frozen release

The selected release uses the score-only control model. These commands write new results to temporary files, preserving the original experiment:

```sh
python training/v4/build.py
python training/v4/run-eval.py dist/models/ntuple-v3.bin - 6410001 5000 /tmp/threes-v4-recheck-v3.jsonl 8
python training/v4/run-eval.py dist/models/ntuple-v4.bin - 6410001 5000 /tmp/threes-v4-recheck-v4.jsonl 8
```

To repeat control training with the same starting weights, data and random seed:

```sh
/tmp/threes-v4-train /tmp/threes-v4-control-retrain control 450 202610011 dist/models/ntuple-v3.bin training/v3/late-pool.txt
```

For the other arms, use modes `a`, `b`, or `c`, seeds 202610012/202610013/202610014 respectively, and `training/v4/fresh-pool.txt`. Mode B also produces a `.five` residual; pass that file as the evaluator's second argument instead of `-`. These CPU-limited retraining runs can produce different final weights; use the archived candidate weights to reproduce the recorded evaluation exactly.
