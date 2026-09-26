# V7: modest failure replay and stronger-search supervision

Starting model: V6 ordinary continued-training candidate, whose previous independent test reached1536 in94.70% of5000 games. That old percentage is not reused as this round's baseline; a new paired holdout evaluates the actual frozen weights. V4 remains the website default unless the prespecified promotion gates pass.

## Locked design

`protocol.json` fixes four arms before inspecting outcomes:

| Arm | Hard resume probability | Strong-search replay |
|---|---:|---|
| r0 |0%|No|
| r10 |10%|No|
| r25 |25%|No|
| r10teacher |10%|Yes|

The remaining episodes start normally. Every arm starts from the same V6-control weights, uses the same training seed and450 process CPU seconds, two-ply action selection, five-step TD(lambda=.5), and lower score learning rates .008/.003/.001 at50%/85% of budget. Finish the last episode after the budget; exclude truncated episodes. Only stage0 changes. On the first1536 afterstate, bootstrap the frozen later-stage score estimate and stop that training episode. Live play continues with unchanged V4 later-stage tables. No probability head replaces the score model.

One anchor afterstate per episode is sampled from fresh baseline trajectories, with the frozen baseline value as target and alpha .001. This is common to all arms and is intended to limit value drift, not a guarantee against forgetting. Its random stream is separate from game and hard-pool sampling. The three non-teacher arms isolate the hard-start ratio. Relative to the starting model, this entire round also changes learning rate, opening distribution and anchor regularization; it does not isolate those effects individually.

The teacher arm adds one weighted teacher afterstate update per episode, with alpha .003/.001/.0003 at the same budget fractions. It uses score-unit regression to stronger-search Q(action) minus immediate merge reward. All legal actions contribute, except afterstates already at1536, whose stage is frozen. This is value-target distillation, not a new classifier policy. Computation spent collecting and generating teacher labels is outside the450-second per-arm training budgets; total end-to-end compute is not equal between methods. CPU-limited retraining is hardware dependent; archived weights define the exact experiment. One training seed per arm.

## Independent training data

Collect2000 normal baseline games on seeds9110001–9112000, stopping at1536 or actual death. Candidate snapshots are failed-game first768 and20/50/100 moves before death, deduplicated within a game. Deterministically select up to256 candidates with seed202609219. All candidates, the subset mapping, seed/turn provenance and simulator snapshots are saved.

Assess each legal first action with8 continuations under the frozen baseline3-ply/24000-node policy. If the two best action rates differ by less than .25, increase all legal actions to24 continuations. A resume state is included if its best action succeeds at least25% of the time and there is at least one success and failure across all actions. This is a heuristic recoverability filter, not proof of solvability or optimality. Hidden deck remainders are reshuffled for simulation; inference sees only the board, public preview and visible-card counts.

Generate teacher action values with depth up to4 and96000 nodes; unfinished iterations fall back atomically to the last completed depth. The frozen baseline provides score leaves. Teacher targets remain ordinary future-score units, never probabilities mixed with points. States whose best action's Wilson lower bound exceeds the worst action's upper bound receive3x replay weight; other included states receive1x. These adaptively selected intervals are a prioritization heuristic, not adjusted statistical evidence about the true best action. The model remains board-only, so labels conditioned on differing public previews/counts are approximated within the same representation.

## Selection and final test

Each arm gets600 full normal games on9310001–9310600. Select highest reach1536 count, then reach6144, then r0/r10/r25/r10teacher order. Freeze the selected weights/hash in selection.json. No holdout-based switching or retraining.

The final test uses5000 paired new seeds9410001–9415000 for the starting baseline, selected candidate and V4. All run to first6144 or actual death, with a6000-move guard; any truncation invalidates the test. Same3-ply/24000 nodes, unlimited offline time. Browser remains160ms soft deadline. Baseline jobs may compute alongside training, but outcomes remain unopened until selection freezes.

Promotion requires a positive paired95% lower bound for reach1536 improvement against BOTH the starting model and V4. For reach6144, require a nonnegative point difference and paired lower95% bound above−1 percentage point against both. Also require browser compatibility. The−1pp margin is a tolerated uncertainty bound, not proof of zero harm. Otherwise V4 stays default and V7 is experimental. Report3072 and conditional6144 given1536 as descriptive diagnostics. Wilson intervals describe rates; paired normal intervals describe per-seed differences, with exact McNemar p-values also retained. Secondary comparisons are not multiplicity adjusted. Shared seeds do not imply identical random outcomes after different actions.

## Reproduce

From the repository root, decompress `training/v6/candidate-control.ntd.gz` to `training/v7/base.ntd`, then:

```sh
python training/v7/build.py
/tmp/threes-v7-verify
python training/v7/collect.py
python training/v7/assess.py
python training/v7/run-arms.py
python training/v7/run-baselines.py
python training/v7/run-final.py
python training/v7/install.py
python training/v7/validate.py
python training/v7/summarize.py
python training/v7/report.py
```

Inference reproduction can use the archived candidate `.ntd.gz` files directly after decompression, avoiding hardware-dependent retraining. Runtime validations cover score-target decomposition, interrupted search, frozen stages, exact native/browser action-depth-node parity, actual worker load/fallback, input behavior and160ms timing using browser code in Node. Timing is not a phone-specific success estimate. Reconstructed bonus-preview rules are not certified identical to the official game.

Research background: [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) and [Policy Distillation](https://arxiv.org/abs/1511.06295). This experiment uses heuristic hard-state sampling and value-target regression; it is not an exact reproduction of either paper.
