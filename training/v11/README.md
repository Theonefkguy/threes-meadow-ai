# V11: policy-only search distillation

A new policy recommends root directions; every original value estimate is unchanged. This is one search-distillation iteration, not a completed multi-round reinforcement-learning system.

## Data and teacher

512 fresh games use the original MCTS with640 simulations per actual move. A separate RNG samples one public state from each of4 rank bands and the last state of failed games. Successes and failures both contribute. Every selected state is labeled by fully completed4ply expectimax, enumerating all chance outcomes, with frozen original score leaves. No teacher node budget truncation is allowed. Soft labels are softmax(Q4/2000), not search visit counts.

The teacher is more exhaustive than the one-step value-based prior being distilled. Prior experiments supported complete4ply over complete3ply, but this round does not assume or establish full4ply beats MCTS640 in complete-game success. This is an explicit limitation of the teacher, not a guaranteed policy-improvement operator.

Source games split by (seed-11410001)%5==0 into validation, otherwise training. D4 augmentation stays within that split. Three initialization/shuffle seeds train50epochs; epoch0 is eligible. Lowest validation policy KL chooses weights before fresh evaluation games. Data and teacher sampling never consume the actual game RNG.

## Policy model

Reuse the small V9 sparse32ReLU architecture with public context: projected board, preview distribution, remaining bag counts, entry locations and frozen board-derived base value. Original baseQ/2000 plus a learned logit residual determines policy priors. Only policy KL supplies gradients (Adam.001, batch16, random D4 augmentation). There is no value loss gradient, no table update, no correction added to leaf values. The unused value output is retained for binary compatibility and verified exactly zero. Legacy Huber/RMSE log fields only measure unchanged base estimates; they do not contribute to optimization or checkpoint selection.

Inference calls the network for legal root actions once per actual move, inside the1msCPU budget. All deeper-node priors and all score evaluations use the original MCTS. No neural calls at1536+. This narrowly scoped root intervention avoids running a new network at every expanded node. It does not measure full-tree policy guidance.

## Evaluation

Baseline and all3 trained seeds run the same512 fresh normal openings, stopping at1536/death, using1msCPU soft budgets. One prespecified primary comparison: mean3 candidates minus baseline, paired by initial seed with t95% interval/test; conditional on these three trained seeds. No test-set model selection. No claim about6144 or browser success. CPU timing is machine dependent and reruns can diverge.

The verification checks80 zero-policy action/visit-count parity cases plus every visited node's exact original value scores and every non-root node's original prior. Trained models must retain zero value heads; base checkpoint SHA remains unchanged.

## Reproduce

From repository root, with C++17, Python, NumPy and SciPy:

```sh
gzip -dc training/v6/candidate-control.ntd.gz > training/v7/base.ntd
g++ -O3 -std=c++17 training/v11/collect.cpp -o /tmp/threes-v11-collect
g++ -O3 -std=c++17 training/v11/train.cpp -o /tmp/threes-v11-train
g++ -O3 -std=c++17 training/v11/bench.cpp -o /tmp/threes-v11-bench
g++ -O3 -std=c++17 training/v11/verify.cpp -o /tmp/threes-v11-verify
/tmp/threes-v11-verify
python training/v11/run.py
python training/v11/analyze.py
python training/v11/report.py
```

The original base is V6 ordinary control (also used in V8–V10), not the website default V4. New policy checkpoints, teacher data, logs, frozen manifest, full game results and scripts are persisted in this directory. No website default promotion without independent evidence and browser/later-stage checks.
