# V10: continue n-tuple TD training

Three conditions: unchanged baseline; ordinary continued TD learning; continued TD with 25% of episode starts sampled from a fixed first-768 pool. Both trained conditions retain exactly the original table architecture and inference code. Three training seeds per condition. No neural residual, teacher distillation, anchor loss or reward shaping.

The fixed base is the V6 ordinary-control model also used in V8/V9, not the website's default V4 weights. Each run receives 120 seconds of process CPU time, with a soft episode-completion boundary. Pool collection is additional one-time preparation and excluded from this training budget. Equal CPU does not imply equal numbers of episodes or TD updates.

The pool contains the first 768 state of each of 128 fresh baseline3ply games that reaches it (127 states). Both successful and subsequently failed games qualify. Sampling is uniform; hidden remaining deck order is reshuffled when resuming. This is 25% of episode starts, not 25% of training steps. The normal-opening majority guards against concentrating entirely on supplied late states. Evaluation always starts normal games.

Training reuses the existing two-ply score behavior policy, five-step truncated lambda return (lambda=.5), and score objective. At1536, the frozen later-stage model supplies a bootstrap boundary; reaching1536 is not substituted for the score objective. The learning-rate schedule is .008/.003/.001 at50%/85% of the CPU budget. The final checkpoint is prespecified; no test-set checkpoint selection. All later-stage model bytes must be unchanged, early bytes changed, and no episode truncated before evaluation starts.

Evaluation uses the exact unchanged V8 MCTS implementation, 1msCPU soft budget per move, stopping at1536/death,512 shared fresh seeds for7 models. Co-primary tests: ordinary-minus-base and mixed-minus-ordinary; average replicas within each game seed, paired t intervals/tests, Holm correction over2 tests. CIs condition on three training seeds; these are not1536 independent initial seeds. CPU timing varies with hardware and can change trajectories. No6144 or browser evaluation, and no automatic website promotion.

## Reproduce from repository root

```sh
gzip -dc training/v6/candidate-control.ntd.gz > training/v7/base.ntd
g++ -O3 -std=c++17 training/v10/train.cpp -o /tmp/threes-v10-train
g++ -O3 -std=c++17 training/v6/collect.cpp -o /tmp/threes-v10-collect
g++ -O3 -std=c++17 training/v8-search/bench.cpp -o /tmp/threes-v10-bench
python training/v10/run.py
python training/v10/analyze.py
python training/v10/report.py
```

`protocol.json` was locked before collection/training. `manifest.json` locks final checkpoint hashes and verifies stage preservation before any new-game evaluation. Compressed checkpoints, full game outcomes, pool and logs are persisted. Intermediate chunks and uncompressed weights are ignored. A rerun can differ due to CPU-timed training and search budgets.
