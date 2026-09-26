# V5: corner potential shaping experiment

Question: does encouraging the maximum tile to occupy a corner improve the probability of making 6144? V4 is the baseline. The original reconstructed Threes rules and score reward are unchanged.

## Locked protocol

See protocol.json. Four arms start with identical V4 weights and the same training RNG seed (202609215), historical 5000-state late pool, stratified starts, 20% recent replay, 2-ply training policy, five-step TD(lambda=.5), and learning-rate schedule. Only the shaping coefficient changes: 0 / 2048 / 8192 / 32768. Each receives 450 process CPU seconds, excluding initial loading and completing the final episode. CPU stopping means episode counts differ; saved checkpoints define the evaluated policies. The pool and baseline hashes are recorded. Hidden deck order is shuffled on restart and is never passed to the policy.

The potential is 1 if a nonterminal afterstate has maximum rank >=13 (3072) and any maximum tile occupies any corner, otherwise 0. This is a late-game-only experiment, not a test of corner rewards throughout the entire game. Four corners are treated symmetrically. No corner is fixed, no action is forbidden, and no extra layout features are bundled in.

Screen: 400 normal games per arm, seeds 7310001–7310400. Freeze highest 6144 count, ties control/weak/medium/strong. Also freeze the best shaping arm, ties weak/medium/strong. New holdout: 5000 normal games each for V4, control and best shaping, seeds 7410001–7415000. This distinguishes additional training from shaping. Do not inspect holdout outcomes before freezing. Only the screen-selected winner is eligible for promotion; no switching candidates after reading holdout. Promotion requires its paired 95% difference lower bound above zero against V4 and browser compatibility. Other pairwise intervals are descriptive and not multiplicity-adjusted. All evaluations use 3 plies / 24000 nodes, original public-information search, and no offline time limit. Stop at first 6144 or actual terminal; report any 6000-move truncations separately.

## Shaping and inference consistency

Gamma is 1, matching the existing undiscounted episodic score learner. For afterstates a and a', the shaped reward is r' = r + c(Phi(a') - Phi(a)). The absorbing terminal potential is zero, including when the final board still has its maximum in a corner. The last target therefore includes r - c Phi(a_last). The trajectory trains through actual terminal, including beyond 6144. Five-step lambda targets use **raw shaped values** U. The browser and native search use original rewards with corrected leaf value V(a) = U(a) + c Phi(a). They never repeatedly add a corner occupancy bonus along search edges.

This distinction is essential: if U accurately equals the original future return minus c Phi, adding back c Phi removes the shaping offset. Reward shaping is a learning aid, not a permanent change to the original objective. The raw tables are initialized identically to V4, so the corrected initial estimate is V4 + c Phi. This intentionally introduces a corner-optimistic initialization. If we instead initialized U exactly to V4 - c Phi, the transformed updates would largely cancel and provide no distinct learning experiment. Native tests verify the telescoping identity, terminal subtraction, and equivalence of all five-step lambda targets under the value transformation.

Potential-based shaping and value initialization are closely related: Wiewiora, *Potential-Based Shaping and Q-Value Initialization are Equivalent*, https://arxiv.org/abs/1106.5267 . Our N-tuple function approximation, partial board representation, finite training and approximate search do not guarantee convergence or improved 6144 probability. The preserved base objective is game score; success at 6144 is the empirical selection metric.

## Validation and reproducibility

Run `python training/v5/build.py`. This compiles C++17 with -O3 -march=native and runs native shaping tests. Re-evaluate any saved candidate with:

```sh
python training/v5/run-eval.py MODEL.ntd COEFFICIENT 7410001 5000 /tmp/threes-v5-recheck.jsonl 8
```

Candidate coefficients are in protocol.json; archived weights are candidate-NAME.ntd.gz. The coefficient is required: using raw shaped weights without the correction is a different, incorrectly evaluated policy. Candidate hashes, training progress summaries, screen results and individual holdout games are retained.

Browser checks compare actual JS directions, depth and node counts to native results, test worker loading/fallback, and compare V4/V5 decisions at the unchanged 160 ms soft limit against unlimited timing on common sampled positions. This is not a phone-specific success-rate measurement. Bonus-preview probabilities remain a reconstruction of the original game, not independently verified official rules. One training seed per arm and one finite compute budget do not establish general superiority or inferiority of corner shaping.
