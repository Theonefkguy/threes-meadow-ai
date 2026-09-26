# V12: counterfactual teacher action audit

Does V11's full4ply score teacher choose better first actions than the original MCTS640, when both subsequently use exactly the same original MCTS640 continuation?

This is a bounded one-step policy-improvement audit. It is not a test of repeatedly using the teacher for the entire game, of V11 student quality, or of6144 success. The base is the same frozen V6 control used in V8–V11, not webpage default V4.

## Design fixed before outcomes

256 independent new source games (11710001–11710256). Uniform reservoir sample of one pre1536 decision state from each game; sampler RNG separate from game RNG. Successful and failed source games both eligible. Unlike V11's four rank bands and additional failure states, this audit weights each source game equally and samples its turns uniformly. Its aggregate effect applies to this sampling distribution.

Teacher: fully completed4ply expectimax with original leaves and no node cutoff. Comparator: unchanged original MCTS640 simulations, depth8, deterministic search seed identical to the source game's policy. Inputs are public board/preview/counts only. The snapshot saves private deck solely to reproduce the simulator; neither decision engine reads it.

If first actions agree, their counterfactual effect is exactly zero because the continuation and seeds would be identical. If they disagree, force each action and run32 matched random futures (11810001+sourceIndex*32+replicate), all using original MCTS640 after the forced move. Each future uniformly reshuffles the remaining hidden deck consistent with observed counts. Current preview probabilities sample bonus identity; future draws use the game engine. Search RNG is independent of game RNG. Matched seeds do not guarantee identical later exogenous events after trajectories diverge.

Every disagreement is included; there is no selection on outcome, Q gap, stage, or perceived severity. Stop each rollout at1536/death. A6000-step guard raises an error rather than silently labeling a truncated game. For each disagreement, the first baseline replicate is run twice and must match exactly (win/moves/reward). These duplicate verification runs are excluded from reported counts.

The primary metric is the mean teacher-minus-base success difference across all256 source states (agreements contribute known zeros). Per-state estimates average32 paired replicates. Paired t95% intervals and tests cluster by source state; repeated continuations are not independent sampled boards. Disagreement-only rates and CI are secondary. Score and rank are exploratory. The exact frozen base hash is checked before/after. Public counts must equal the remaining simulator deck counts.

## Reproduce

From repository root with C++17, Python, NumPy and SciPy:

```sh
gzip -dc training/v6/candidate-control.ntd.gz > training/v7/base.ntd
g++ -O3 -std=c++17 training/v12-teacher/experiment.cpp -o /tmp/threes-v12-experiment
python training/v12-teacher/run.py
python training/v12-teacher/analyze.py
python training/v12-teacher/report.py
```

The search uses a very large time ceiling and stops at640 simulations for deterministic action selection. CPU timings are observational, not budget-matched evidence. Persisted snapshots, public action metadata, all paired rollout outcomes, protocol, summary and code allow inspection. Intermediate chunks are ignored. No weights or website files are modified.
