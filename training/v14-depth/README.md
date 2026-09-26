# V14: complete four-ply versus five-ply expectimax pilot

Fixed16 paired new normal openings, seeds12010001–12010016. Freeze the V6 ordinary-control n-tuple checkpoint used by V8–V13. Original V7 iterative expectimax enumerates every legal action and public chance outcome. INT_MAX node budget; every actual move must report completed==requested depth. No time cutoff, sampling or beam pruning. Search depends only on public information and does not consume the environment RNG.

Stop on1536/death. A6000move guard raises an error, not a losing outcome. Both arms use the same initial seeds, but diverging decisions do not guarantee identical subsequent spawn events. This is a16-pair feasibility pilot, not a well-powered success-rate comparison. Report paired outcomes, exactMcNemar test, exact per-arm binomial intervals and actual process CPU cost (game and move averages). Timing ratios reflect the visited gameplay distributions, not identical-board microbenchmarks.

The complete5ply arm is different from selectively sampling some paths todepth8 in MCTS. There is no equal-time constraint; the experiment tests spending more computation while freezing the value model. There is no model training, browser promotion or6144 evaluation.

Reproduce from repo root with C++17/Python/NumPy/SciPy:

```sh
gzip -dc training/v6/candidate-control.ntd.gz > training/v7/base.ntd
g++ -O3 -std=c++17 training/v14-depth/bench.cpp -o /tmp/threes-v14-bench
python training/v14-depth/run.py
python training/v14-depth/analyze.py
python training/v14-depth/report.py
```

Eight process workers execute the32 games. Protocol fixed before running; no sample-size increase based on observed wins. Complete per-game outcomes, total nodes, timing summaries and source are persisted. Intermediate chunk/progress files are ignored. Baseline hash checked before and after the run. The website's V4 default is not the frozen value checkpoint used here.
