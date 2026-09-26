# V6: improving reach1536 without sacrificing reach6144

This round changes only the pre-1536 stage. V4 remains the frozen late-game teacher and handoff policy. The objective is higher success from real openings, not a high win rate on a selected resume pool.

## Locked experiment

See `protocol.json`, written before collection outcomes. Three arms receive 450 process CPU seconds each (plus completion of the last episode), the same seed and N-tuple capacity:

- **control:** ordinary score TD, resumed from first-768 states of all collection games.
- **hard:** identical score TD, resumed from recoverable failure states instead.
- **probability:** the same hard pool, 90 CPU seconds of supervised initialization and the remaining budget of goal TD. Stage 0 stores logits of reaching1536, rather than future score. This is a bundled objective/initialization intervention, not a pure reward ablation.

All arms gradually reduce resume probability (.8 / .5 / .2 at .5/.85 of the total CPU budget), returning to fresh openings. Sampling is uniform within each arm's pool. Only stage0 is updated. Both later stage tables are checked byte-for-byte against V4.

Training is two-ply, five-step TD(lambda=.5). Score step rewards are next spawn points plus next merge reward. The first1536 afterstate is a bootstrap boundary with frozen V4 score value. For the probability arm, ordinary rewards are zero, first1536 is absorbing value1, and earlier death is0; five-step targets stay within [0,1]. Sigmoid updates use cross-entropy gradients normalized by the squared multiplicity of active D4-symmetric features. Warmstart samples V4 afterstates uniformly with final success/failure labels and alpha .1. Initial probability .9. There is no per-move corner reward in this round.

Score learning rates: .02/.008/.003; goal learning rates: .1/.04/.015 at .5/.85 CPU fractions. Loading is outside the budget; warmstart is included. Hidden remaining decks are shuffled when resuming, and only public preview/counts are passed to inference. Different policy trajectories consume random numbers differently even with paired seeds.

## Fresh data

Collect 2000 V4 games (seeds8110001–8112000), stopping at1536 or death. Record first768 states for the ordinary pool. For failed games, record first768 where present and states20/50/100 moves before failure (deduplicated within a game). Each candidate gets4 simulated continuations per legal forced first action, followed by V4 three-ply. Select candidates with at least one success and one failure. This noisy filter is not proof of solvability or impossibility. Assessment RNG uses2026100000 + candidateIndex*128 + direction*16 + continuationIndex.

The teacher set records afterstates every20 moves, or every5 moves while the current maximum is768, with eventual reach1536 labels. It never includes reached1536 afterstates. Full snapshots include simulator internals to reproduce resume training; policy search cannot access them. `sources.jsonl` maps pool entries to collection seed/turn; `hard-indices.json` maps the hard pool to assessment and source entries.

## Screen, freeze, test

Each arm gets400 new complete games (8310001–8310400). Highest reach1536 count wins; ties break by reach6144 then control/hard/probability order. `selection.json` freezes weights/hash before any holdout outcomes are inspected.

The final test uses5000 new paired seeds8410001–8415000 for V4, control and selected candidate (reuse control if selected). All begin at normal openings and continue to first6144 or actual death, with a6000-move guard; any truncation invalidates the test. Depth3,24000 nodes, unlimited offline wall time; actual browser retains a160ms soft deadline. V4 holdout may compute during training, but its results are not inspected until selection is frozen.

Promotion requires a positive paired95% lower bound for reach1536 difference; nonnegative reach6144 point difference and its lower bound above−1 percentage point; and runtime compatibility. The6144 gate allows uncertainty within a prespecified margin; it is not proof of no harm. Also report full reach3072 and conditional progression after1536. No holdout-driven retraining, arm substitution or budget extension.

## Reproduce

From repository root:

```sh
python training/v6/build.py
/tmp/threes-v6-verify
python training/v6/collect.py
python training/v6/run-arms.py
python training/v6/run-eval.py dist/models/ntuple-v4.bin 0 8410001 5000 training/v6/holdout-v4.jsonl 5
python training/v6/run-final.py
python training/v6/install.py
python training/v6/summarize.py
python training/v6/validate.py
python training/v6/report.py
```

Models and all individual game records are archived. CPU-limited retraining is hardware-dependent; decompress candidate `.ntd.gz` files to reproduce inference for the exact weights. Stage0 tables occupy bytes16 through2097167; the remainder must equal V4. Model format remains NTD1 with6MiB of weights; probability inference uses the same capacity, not an additional downloaded model.

Validation includes native goal/death/boundary/multiplicity/symmetry tests, exact browser/native action/node/depth comparisons for score and probability candidates, worker loading/fallback tests and game input tests. Timing runs actual browser search code in Node, not a phone. Bonus-preview mechanics are reconstructed; the results do not certify equality with the official game's distribution. One training seed and short fixed budget are insufficient to judge the general merit of goal training.
