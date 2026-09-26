# V15 late-stage depth experiment

This fixed pilot compares continuous complete4ply and complete5ply expectimax from128 independently sourced first-768 snapshots. Four matched hidden-deck reshuffles per snapshot produce1024 total continuations. Each arm continues to3072/death and also records reaching1536.

The independent unit is the source snapshot. Four repetitions are averaged within snapshot. Primary endpoint: depth5-minus-depth4 probability of reaching1536. Secondary endpoints: reaching3072 and the difference between the3072 and1536 treatment effects. Paired t intervals/tests use128 snapshot clusters. Results are conditional on this frozen value model and snapshot distribution.

All chance branches in search are enumerated; node budget is INT_MAX and every move must report the requested completed depth. Search does not read the hidden deck. Before each continuation, the remaining ordinary deck is reshuffled consistently with public counts. Arms share rollout seeds, while divergent trajectories need not experience identical later events.

The runner is resumable both between tasks and inside a continuation: valid per-task result files are skipped, while an atomic per-task checkpoint preserves the full game, RNG, public counts, and accumulated metrics every10 moves. It writes `status.json` every16 completed tasks, then merges raw results and runs analysis/report automatically. The fixed protocol forbids outcome-driven stopping or extension.

Run from repo root:

```sh
g++ -O3 -std=c++17 training/v15-late-depth/continue.cpp -o /tmp/threes-v15-continue
python training/v15-late-depth/run.py
```

The source pool and frozen model were already committed. Intermediate progress files are ignored; raw completed task results are retained so an interrupted run can resume.
