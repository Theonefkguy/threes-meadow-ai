# V7 diagnostic experiments

This directory diagnoses the previous unsuccessful reach-1536 improvement. It is not a new deployed policy.

## Reproduction

Run from the repository root. Requires C++17, Python3 and scipy.

1. Restore `training/v7/base.ntd` from `training/v6/candidate-control.ntd.gz` if needed; expected SHA256 is `d863b01eb0b958237e5ba9833da38c8f57ec833c983e4add1a196a515cbff590`.
2. Compile `games.cpp` with `g++ -std=c++17 -O3 -march=native training/diagnosis-v7/games.cpp -o /tmp/threes-diagnosis-games`.
3. Run `python training/diagnosis-v7/run-games.py`. This launches eight native workers for 512 common fresh seeds per policy, stopping at first1536/death. Full four-ply is substantially more expensive than the deployed search. All requested depths must complete for exact modes. Protocol fixed before reading game success outcomes.
4. Run `python training/diagnosis-v7/run-fit.py`. This audits original candidates, then fits teacher labels with a source-game-disjoint split and three shuffle seeds. Exact and D4-equivalent afterstates cannot leak across the split. Diagnostic fitting uses alpha=.01, weighted teacher examples and epochs1/10/100. All later-stage weights are checked bit-identical to the baseline.
5. Run `python training/diagnosis-v7/run-cv.py` for a secondary five-fold sensitivity check. This was added after seeing the first split's fit/generalization gap; it is not an independent confirmatory study. All examples receive an out-of-fold prediction per epoch and seed.
6. Compile/run `snapshots.cpp` to compare original limited search with completed3/4-ply on the256 old assessed positions. This reuses noisy8/24-rollout baseline continuations; their success fractions are exploratory and are not full teacher performance.
7. Run `python training/diagnosis-v7/report.py` to regenerate `results.json`.

## Interpretation boundaries

- Teacher values express expected future **score**, not the probability of reaching1536. Prediction RMSE alone cannot establish game improvement.
- Original V7 candidates were trained on all951 teacher examples; their label audit is **in-sample**, regardless of the later diagnostic split. Only the newly supervised fits exclude their held-out games.
- The three fit seeds vary shuffle order, not independent RL trajectories. No new online RL-policy training occurs here.
- A large train/test fit gap supports limited generalization on these targets. It does not uniquely separate sparse data, feature representation, target noise and distribution shift.
- A512-seed paired game test is intended to identify large differences, not prove equality or exclude~1percentage-point gains.
- Search leaves and score objective are held fixed. Full4-ply is a diagnostic computation, not a browser-ready replacement.
- Existing V4 default and website assets are unchanged. Subsequent decisions should use a fresh test set.

## Secondary interference probe

Compile/run `interference.cpp`. Start each seed from the100-epoch teacher fit above. Collect100 fresh normal games with the frozen baseline2-ply policy (seeds9710001–9710100); fix theirapproximately51,000 five-step lambdaTD targets. Replay the identical TD sequence in each dose arm for five passes at alpha=.003, inserting teacher updates at alpha=.001 in count ratios0/0.21%/2%/10%. All arms use the same ordinary TD data and order within seed. The teacher labels exclude the20 heldout source games. This directly tests update interference under fixed replay, not online RL-policy improvement; no1536 success claims follow from this probe. It was added as a secondary diagnostic after initial fit results.

## Secondary input/target context check

Compile/run `context.cpp`. For each old assessed position with a normal preview, restore pre-draw public counts by adding back the displayed card. If at least two normal card types remain possible, recompute completed3-ply teacher targets for each possible normal preview and corresponding remaining counts. Keep the board, chosen action and resulting afterstate fixed. Weight contexts by their conditional normal-draw probabilities. A board-only value model must give exactly the same prediction across these contexts, even when the conditional teacher targets differ. The within-afterstate weighted variance therefore gives a lower bound on prediction MSE **on this synthetic context dataset**, not the original951 labels or game performance. This reveals an information bottleneck, not a claim that the deployed search ignores previews (it does use them). Averaging conditional targets appropriately or adding context features are different possible remedies and require further tests.

The first interference pilot used seeds9610001–9610100. A provenance check found that three of those seeds had appeared in an older, different-policy code-parity check (not training). Its output is retained as `interference-pilot-9610001.jsonl`; the reported interference result is a full replication on unused seeds9710001–9710100, with all settings unchanged. To reproduce the pilot, pass9610001 and an output path to the interference executable.
