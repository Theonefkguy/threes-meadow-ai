# V6 early-stage results

Screen-selected arm: **hard**. Policy: **score**. Default promotion: **False**.

5000 shared new seeds8410001–8415000 per policy, from ordinary openings until first6144 or actual death. All later-stage weights remain exactly V4. Native3-ply search,24000 nodes, no wall-time deadline. No truncated games. The selected arm was frozen before inspecting the holdout.

| Policy | 1536 count | Reach1536 | Reach3072 | Reach6144 | 6144 count | 6144 given1536 |
|---|---:|---:|---:|---:|---:|---:|
| V4 baseline | 4647/5000 | 92.94% | 65.46% | 14.78% | 739/5000 | 15.90% |
| Ordinary early-stage continued training | 4735/5000 | 94.70% | 66.94% | 15.08% | 754/5000 | 15.92% |
| hard | 4695/5000 | 93.90% | 66.92% | 16.28% | 814/5000 | 17.34% |

| Paired comparison (second minus first) | Target | Difference, pp | Paired95% interval, pp | Exact McNemar p |
|---|---:|---:|---:|---:|
| v4_vs_control | 1536 | 1.76 | 0.81 to 2.71 | 0.00031 |
| v4_vs_control | 3072 | 1.48 | -0.37 to 3.33 | 0.12279 |
| v4_vs_control | 6144 | 0.30 | -1.09 to 1.69 | 0.69318 |
| v4_vs_selected | 1536 | 0.96 | -0.01 to 1.93 | 0.05818 |
| v4_vs_selected | 3072 | 1.46 | -0.38 to 3.30 | 0.12484 |
| v4_vs_selected | 6144 | 1.50 | 0.10 to 2.90 | 0.03864 |
| control_vs_selected | 1536 | -0.80 | -1.71 to 0.11 | 0.09441 |
| control_vs_selected | 3072 | -0.02 | -1.88 to 1.84 | 1.00000 |
| control_vs_selected | 6144 | 1.20 | -0.20 to 2.60 | 0.09883 |

Primary comparison **v4_vs_selected**. Early improvement gate: **False**. End-to-end6144 gate: **True**. The latter permits a95% lower bound above−1pp, while requiring a nonnegative point difference; this does not prove zero harm. Secondary comparisons are descriptive and not multiplicity-adjusted. Per-policy Wilson intervals and failure maxima are in evaluation.json.

## Equal-CPU training and independent screening

| Arm | Episodes | CPU seconds | Warmstart updates |1536/400 |3072/400 |6144/400 |
|---|---:|---:|---:|---:|---:|---:|
| control | 169284 | 450 | 0 | 376 | 266 | 71 |
| hard | 215119 | 450.003 | 0 | 381 | 266 | 69 |
| probability | 323774 | 450.002 | 73507030 | 26 | 18 | 6 |

Collected2000 fresh V4 trajectories; 126 failed before1536. Assessed498 candidate snapshots, selected335 with mixed continuation outcomes. No holdout states entered training.

Control and hard arms differ only in resume-pool distribution. Probability additionally changes the objective and spends90 of450 CPU seconds on supervised initialization; this is not a pure reward-only comparison. Negative results would concern this implementation/budget, not all goal-directed reinforcement learning. Exact archived weights, one training seed per arm.

## Runtime checks

{
  "native": "PASS normalized probability updates, frozen stages, D4 symmetry, absorbing goal/death, lambda boundary, score handoff, probability-only backup",
  "model": "{\"passed\":true,\"arms\":3,\"checks\":[\"finite weights\",\"bitwise frozen V4 stages\",\"bounded sigmoid\",\"absorbing1536\",\"terminal\",\"V4 handoff\"]}",
  "parity-selected": "{\"model\":\"training/v6/selected.ntd\",\"probabilityMode\":0,\"positions\":87,\"exact\":[\"direction\",\"nodes\",\"depth\"]}",
  "parity-probability": "{\"model\":\"training/v6/run-probability/checkpoint.ntd\",\"probabilityMode\":1,\"positions\":25,\"exact\":[\"direction\",\"nodes\",\"depth\"]}",
  "worker": "PASS 35 actual worker cases",
  "input": "✔ drag cancel, committed swipe, hidden candidates, blocked swipe and restart race (40.535356ms)\n✔ interrupted gestures always clear preview without drawing a card (22.271316ms)\n✔ release outside the board commits exactly once (1.396599ms)\n✔ late capture loss from an older touch cannot cancel the new touch (5.375264ms)\n✔ focus loss during a committed slide preserves the pending move (1.314387ms)\n✔ AI starts from the current game, moves repeatedly, and stops without accepting stale replies (1.764865ms)\n✔ manual input, new game, backgrounding and worker failure stop AI (8.284954ms)\n✔ AI automatically stops at game over (1.729953ms)\n✔ manual play is counted before AI takeover and restart resets the visible memory (17.522941ms)\n✔ switching learned policy stops old work, and failed model load visibly falls back (2.057688ms)\n✔ V2 takeover sends only visible information and load fallback preserves V1 choice (1.273617ms)\n✔ V3 load fallback keeps subsequent turns on V2 (6.805152ms)\n✔ V4 load fallback keeps subsequent turns on V3 (1.262661ms)\n✔ V5 load fallback keeps subsequent turns on V4 (5.404647ms)\n✔ V6 load fallback keeps subsequent turns on V4 (1.326716ms)\nℹ tests 15\nℹ suites 0\nℹ pass 15\nℹ fail 0\nℹ cancelled 0\nℹ skipped 0\nℹ todo 0\nℹ duration_ms 305.030281",
  "timing": "{\n  v4: {\n    positions: 97,\n    meanMs: 32.369860608247414,\n    p95Ms: 71.33782999999858,\n    maxMs: 97.55938100000003,\n    depths: [ 0, 0, 2, 95 ],\n    changedFromUnlimited: 0\n  },\n  v6: {\n    positions: 97,\n    meanMs: 31.794625886597863,\n    p95Ms: 71.65447300000051,\n    maxMs: 104.90954099999908,\n    depths: [ 0, 0, 2, 95 ],\n    changedFromUnlimited: 0\n  },\n  environment: 'Actual browser JS search in Node on this machine, alternating order; same leaf cache optimization for both models. Not a phone-specific success estimate.',\n  compatible: true\n}"
}

| Policy | Positions | Mean ms |p95 ms |Changed from unlimited at160ms |
|---|---:|---:|---:|---:|
| v4 | 97 | 32.37 | 71.34 | 0 |
| v6 | 97 | 31.79 | 71.65 | 0 |

Actual browser search code in Node on this machine, not a phone success estimate. Reconstructed bonus-preview rules are not independently proven identical to the official app. Frozen selected SHA256 `f093e577a568db9d0861cc290ae20fccb4755bb59dd2fd3654fa520ca7c4f77a`. See training/v6/README.md for objective boundaries, sampling, reproduction, and limitations.
