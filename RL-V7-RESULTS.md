# V7 early-stage experiment results

Selected arm: **r10teacher**. Default promotion: **False**.

## New5000-game test

Paired seeds9410001–9415000, normal openings to first6144 or actual death. No truncated games. Same3-ply/24000 nodes, unlimited offline time. Model selected and frozen before holdout inspection.

| Policy |1536 count |Reach1536 |Reach3072 |Reach6144 |6144 count |6144 given1536 |
|---|---:|---:|---:|---:|---:|---:|
| V6 ordinary continued-training baseline | 4716/5000 | 94.32% | 65.98% | 15.36% | 768/5000 | 16.28% |
| V7 r10teacher | 4710/5000 | 94.20% | 67.66% | 14.56% | 728/5000 | 15.46% |
| V4 website default | 4666/5000 | 93.32% | 65.14% | 14.88% | 744/5000 | 15.95% |

| Comparison (selected minus baseline) |Target |Difference, pp |Paired95% interval, pp |Exact McNemar p |
|---|---:|---:|---:|---:|
| baseline_vs_selected | 1536 | -0.12 | -1.02 to 0.78 | 0.82809 |
| baseline_vs_selected | 3072 | 1.68 | -0.12 to 3.48 | 0.07088 |
| baseline_vs_selected | 6144 | -0.80 | -2.17 to 0.57 | 0.26612 |
| v4_vs_selected | 1536 | 0.88 | -0.07 to 1.83 | 0.07459 |
| v4_vs_selected | 3072 | 2.52 | 0.68 to 4.36 | 0.00763 |
| v4_vs_selected | 6144 | -0.32 | -1.67 to 1.03 | 0.66423 |

Promotion gates: {"baseline_vs_selected": {"early": false, "late": false}, "v4_vs_selected": {"early": false, "late": false}}. The primary goal is improved1536 versus the starting model. Promotion additionally checks V4 and a−1pp6144 uncertainty margin with nonnegative6144 point differences. Other reported comparisons are descriptive, not multiplicity adjusted. See evaluation.json for all rate intervals.

## Equal training budget and separate600-game screen

| Arm |Hard-start ratio |Teacher replay |Episodes |CPU seconds |1536/600 |3072/600 |6144/600 |
|---|---:|---|---:|---:|---:|---:|---:|
| r0 | 0.00% | False | 100155 | 450.002 | 558 | 382 | 103 |
| r10 | 10.00% | False | 111617 | 450.006 | 570 | 406 | 93 |
| r25 | 25.00% | False | 126617 | 450.002 | 567 | 406 | 90 |
| r10teacher | 10.00% | True | 109823 | 450 | 574 | 401 | 89 |

Fresh data: {"collected": 2000, "failedBefore1536": 101, "candidates": 396, "assessed": 256, "hard": 252, "extendedTo24": 245, "clearSpread": 12, "teacherDepths": {"1": 0, "2": 0, "3": 212, "4": 44}, "teacherExamples": 951, "anchorExamples": 117464}.

All arms preserve later V4 tables, use lower learning rates and common frozen-value anchors. Ratios isolate replay-distribution effects; r10teacher versus r10 isolates added value-target distillation within the fixed training budget. Teacher-generation compute is additional and outside that budget. The stronger teacher predicts score, not necessarily1536 success. A negative result does not establish that all curriculum learning or distillation is ineffective.

## Runtime validation

```json
{
  "native": "PASS root action targets, reward subtraction, interrupted-iteration atomicity, two-ply policy agreement, finite teacher values, frozen later stages",
  "stageIsolation": "All four candidates retain byte-identical later stage tables.",
  "parity": "{\"model\":\"training/v7/selected.ntd\",\"probabilityMode\":0,\"positions\":87,\"exact\":[\"direction\",\"nodes\",\"depth\"]}",
  "worker": "PASS 40 actual worker cases",
  "input": "✔ drag cancel, committed swipe, hidden candidates, blocked swipe and restart race (52.328738ms)\n✔ interrupted gestures always clear preview without drawing a card (18.060861ms)\n✔ release outside the board commits exactly once (2.340166ms)\n✔ late capture loss from an older touch cannot cancel the new touch (1.569214ms)\n✔ focus loss during a committed slide preserves the pending move (1.446653ms)\n✔ AI starts from the current game, moves repeatedly, and stops without accepting stale replies (9.88408ms)\n✔ manual input, new game, backgrounding and worker failure stop AI (4.526565ms)\n✔ AI automatically stops at game over (1.456347ms)\n✔ manual play is counted before AI takeover and restart resets the visible memory (30.600846ms)\n✔ switching learned policy stops old work, and failed model load visibly falls back (1.752406ms)\n✔ V2 takeover sends only visible information and load fallback preserves V1 choice (6.89688ms)\n✔ V3 load fallback keeps subsequent turns on V2 (5.737739ms)\n✔ V4 load fallback keeps subsequent turns on V3 (1.319905ms)\n✔ V5 load fallback keeps subsequent turns on V4 (1.320536ms)\n✔ V6 load fallback keeps subsequent turns on V4 (5.404845ms)\n✔ V7 load fallback keeps subsequent turns on V4 (1.343059ms)\nℹ tests 16\nℹ suites 0\nℹ pass 16\nℹ fail 0\nℹ cancelled 0\nℹ skipped 0\nℹ todo 0\nℹ duration_ms 324.622743",
  "timing": "{\n  v4: {\n    positions: 97,\n    meanMs: 20.45388176288657,\n    p95Ms: 48.38684599999942,\n    maxMs: 74.87819800000034,\n    depths: [ 0, 0, 1, 96 ],\n    changedFromUnlimited: 0\n  },\n  v7: {\n    positions: 97,\n    meanMs: 20.38887781443298,\n    p95Ms: 44.54132200000004,\n    maxMs: 94.59105900000031,\n    depths: [ 0, 0, 1, 96 ],\n    changedFromUnlimited: 0\n  },\n  environment: 'Actual browser JS search in Node on this machine, alternating order; same leaf cache optimization for both models. Not a phone-specific success estimate.',\n  compatible: true\n}"
}
```

| Policy |Positions |Mean ms |p95 ms |Changed from unlimited at160ms |
|---|---:|---:|---:|---:|
| v4 | 97 | 20.45 | 48.39 | 0 |
| v7 | 97 | 20.39 | 44.54 | 0 |

Browser search code measured in Node on this machine, not a phone-specific success rate. Reconstructed game rules. Single training seed and budget. Selected model SHA256: `575283e30c0b64ea1da70c21eb7ff798a7c3528d80a9cfe4f37c531a71291e18`. Code, exact weights, source data and individual-game results are retained under training/v7.
