# V13: no-tree student plus outcome-based on-policy RL

The V11 student is evaluated with no tree search, then trained from its own stochastic games and actual1536/death outcomes. This is an on-policy REINFORCE pilot, not relabeling model predictions as ground truth. The policy still uses the frozen n-tuple value as a feature and fixed logit component; it is not a neural-only controller.

## Prespecified setup

Initial student is V11 seed202609252 selected by lowest original validation policyKL, not by game scores. Initial student and original MCTS1ms run512 common fresh opening seeds11910001–11910512. Each of3 training seeds then plays8192 normal-opening games in4 rounds of2048. No search or teacher during data generation. Behavior samples softmax(logits/.1); game RNG, action RNG and replay RNG are separate. Success at1536 yields1, death0. No intermediate merge reward, probability shaping or curriculum resets.

Batch64 complete games under frozen batch weights. Gradient is REINFORCE with terminal outcome minus a per-rank-band EMA success baseline from PREVIOUS batches (decay.9, initial.5). Gradient sums the entire trajectory and averages64 episodes. Baseline is action-independent at the current state. No discount. Global gradient clip1, Adam3e-5 with beta.9/.999, eps1e-8. Four intermediate checkpoints are fixed at2048/4096/6144/8192 games; there is no outcome-based checkpoint selection.

Replay regularization:64 old V11 TRAINING states per batch, .02KL(initialPolicy||currentPolicy) atT1. Only board/public context is reused; old teacher Q labels are discarded. This anchors earlier behavior but provides no new improvement signal. Fresh terminal game outcomes supply the RL gradient. All n-tuple weights and unused neural value output remain unchanged. Finite-difference checks verify the sampled log-probability gradient including temperature and reward advantage.

Evaluation is greedy at every checkpoint and never samples exploration actions. All checkpoints and baseline use the same512 independent evaluation seeds. Primary test compares mean3 final models to the initial direct student, paired by game seed with t95% interval, conditional on3 training seeds. Round1–3 are descriptive and not selected for promotion. Repeated seeds are not independent additional openings. Training win rates use stochastic behavior and must not be compared directly with greedy evaluation rates.

The MCTS comparison shows the cost/quality tradeoff; it does not provide the same compute budget to both models. CPU means measure decision calls only, excluding game environment steps. All episodes stop at1536/death and a6000-step overrun is an error. No6144/browser test or default website promotion.

## Reproduce

From repo root, C++17 plus Python/NumPy/SciPy:

```sh
gzip -dc training/v6/candidate-control.ntd.gz > training/v7/base.ntd
g++ -O3 -std=c++17 training/v13-selfplay/train.cpp -o /tmp/threes-v13-train
g++ -O3 -std=c++17 training/v13-selfplay/bench.cpp -o /tmp/threes-v13-bench
g++ -O3 -std=c++17 training/v13-selfplay/verify.cpp -o /tmp/threes-v13-verify
python training/v13-selfplay/run.py
python training/v13-selfplay/diagnostics.py
python training/v13-selfplay/analyze.py
python training/v13-selfplay/report.py
```

Training is fixed-game, rather than fixed-time, and deterministic for a seed/compiler. Greedy actor evaluation is deterministic. MCTS timing can change its trajectory. Code, protocol, all rounds' weights, training curves, evaluation outcomes and initial/base hashes are persisted. Intermediate chunk files are ignored. This directory depends on the committed V11 model and old training-state file.

Posthoc amendments: all-zero primary successes make the prespecified paired-t variance degenerate. Analysis reports no t-test p-value and instead uses a conservative Bonferroni combination of exact one-sided binomial bounds for initial success and any-final-student success per game-seed cluster. Direct-table and existing-two-ply controls were added for diagnosis after observing zero student success, not as confirmatory endpoints. Their outcome files and runner are retained.
