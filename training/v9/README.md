# V9: public-context × policy-head search distillation

This experiment trains 12 small residual networks, not a completed iterative on-policy RL system. See `../../RL-V9-ABLATION.md` for results and limits. The exact protocol was written before evaluation; checkpoint hashes were frozen before fresh games.

## Reproduce

Run from the repository root. Requires Python 3, NumPy, SciPy and a C++17 compiler. Restore the frozen base if absent:

```sh
gzip -dc training/v6/candidate-control.ntd.gz > training/v7/base.ntd
g++ -O3 -std=c++17 -pthread training/v9/collect.cpp -o /tmp/threes-v9-collect
g++ -O3 -std=c++17 -pthread training/v9/train.cpp -o /tmp/threes-v9-train
g++ -O3 -std=c++17 -pthread training/v9/bench.cpp -o /tmp/threes-v9-bench
g++ -O3 -std=c++17 -pthread training/v9/verify.cpp -o /tmp/threes-v9-verify
/tmp/threes-v9-verify
python training/v9/collect.py
python training/v9/train.py
python training/v9/evaluate.py
python training/v9/analyze.py
```

The committed teacher data and all 12 `best.net` checkpoints allow skipping collection/training. CPU-timed searches are hardware dependent: a rerun need not reproduce exact game outcomes. Seeds isolate game RNG from teacher sampling/search RNG, but diverging trajectories do not share identical exogenous spawn events.

## Data and labels

1200 source games; random per-game reservoirs in four board-rank bands plus terminal-adjacent failures. Both successes and failures are included. Full 4-ply expectimax provides action scores; policy labels are a temperature-2000 softmax of these scores, **not MCTS visit counts**. 960 source games train, 240 validate. Augmented boards and their entry masks remain in their source split. No D4-equivalent board occurs in both splits.

## Models and controls

Four arms, three initialization/shuffle seeds each. All use a 32-ReLU hidden layer with separate value and policy outputs. Value-only arms do not train the policy output; their priors come from corrected value scores. Context arms add public preview probabilities, remaining bag counts and permitted spawn locations to the projected board. All arms receive the same frozen board-derived value feature.

Value residuals use Huber loss, normalized by 8000. Joint arms add 0.2 times policy KL. Training lasts 30 epochs; epoch 0 is eligible. Every arm selects its checkpoint by validation Huber + 0.2 KL before fresh game evaluation. Joint heads can compete for the shared hidden representation; this factorial intervention measures the entire joint-training choice, not an isolated policy output detached from value learning.

Inference adds the residual to frozen values only before rank 12. At or above 1536, original values are used. Changing early play can still affect later success; this experiment does not estimate 6144 success.

## Evaluation

All 12 networks and unchanged MCTS run the same 512 fresh initial seeds, 1 ms CPU soft budget per move, depth limit 8, stopping on 1536/death. This is 6656 runs on 512 common initial seeds, not 6656 independent initial conditions. Training seeds share teacher data.

Co-primary factorial effects average the three trained replicas within each initial seed, then use a paired t interval/test across 512 game-seed clusters. Holm adjusts the two main-effect tests. These intervals are conditional on the three fitted seeds and do not measure uncertainty over arbitrary training runs. Individual comparisons against baseline are exploratory. No checkpoint is selected using evaluation wins. No website default promotion is part of this experiment.
