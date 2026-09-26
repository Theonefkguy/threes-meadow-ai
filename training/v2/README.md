# V2 experiment files

See ../../RL-V2.md for methods, reproduction and limits.

- protocol.json: primary endpoint, disjoint seed ranges, search budgets and candidate selection rule.
- common.h / train-score.cpp: real-state curriculum, hard-state replay, multi-step score learning.
- goal.h / train-goal.cpp: sigmoid target head and score-guided training, with optional public context.
- goal-pure.h / train-goal-pure.cpp / failed-pilots.json: unsuccessful initial pure-probability trials.
- search.h / benchmark.cpp: deterministic native equivalent of the browser search; evaluation stops at goal or terminal.
- run-*/progress.jsonl: actually executed training logs; selected checkpoint episode count is recorded separately from any subsequent unused computation.
- screen-*.jsonl: 200 fresh-opening games per candidate, used only for selection.
- selection.json: frozen model hashes and screening selection, written before holdout outcomes are read.
- holdout-blinded/*.jsonl: raw independent test records. The name documents that the baseline was generated before candidate selection without inspecting outcomes; it is not an access-control mechanism.
- evaluation.json: final 1,000-game-per-policy summary with intervals and paired test.
- verify-search.mjs / verify-goal.mjs: exact direction/depth/node parity with browser JavaScript, using probe records from benchmark.cpp.
- verify-learning.cpp: independent numerical multi-step targets, goal/terminal boundaries, sigmoid update and context index bounds.
- timing.json: JavaScript timing on the execution machine, not mobile-browser measurement.

Checkpoints in run-* and native executables are ignored. Selected production weights live in ../../dist/models. Final alternative candidate snapshots are compressed in this directory when retained for research reproduction. Logs are not fabricated from estimated throughput.
