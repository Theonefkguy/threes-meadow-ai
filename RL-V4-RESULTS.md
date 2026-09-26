# Threes V4 research results

Selected candidate: **control**. Promoted as default: **True**.

## Independent 5000-game test

New seeds 6410001–6415000; same reconstructed engine, 3-ply expectimax and 24000-node budget; no wall-clock limit offline. First 6144 or actual terminal, no truncations. Selection/frozen hashes preceded inspection of this test. These are fresh estimates, not the previous test's results.

| Metric | V3 | V4 |
|---|---:|---:|
| Reached 6144 | 665/5000 | 744/5000 |
| 6144 rate | 13.30% | 14.88% |
| 6144 Wilson 95% CI | 12.39%–14.27% | 13.92%–15.89% |
| Reached 3072 | 65.98% | 65.98% |
| 6144 given 3072 | 20.16% | 22.55% |

Paired difference 1.58 percentage points, approximate paired 95% CI [0.34, 2.82] pp. Exact two-sided McNemar p=0.014026912465728153. V3-only successes 465; V4-only 544. A shared seed does not guarantee identical subsequent random outcomes after policies diverge.

## Compute-matched screening

Each arm received approximately 450 process CPU seconds, excluding initial loading; final episode allowed to finish. Shared new-data collection (1000 V3 games yielding 4854 states from 680 source games) is additional. Training times include policy search and learning overhead; episode counts intentionally differ.

| Arm | Change | Training episodes | CPU seconds | 6144 / 400 screening games |
|---|---|---:|---:|---:|
| control | Continue V3 on historical late pool, 2-ply | 279890 | 450 | 56/400 |
| a | Fresh pool + 5% 3-ply episodes | 53550 | 450 | 40/400 |
| b | A data/policy + frozen V3, five-cell residual | 43000 | 450.42 | 39/400 |
| c | A data/policy + Temporal Coherence | 52043 | 450 | 44/400 |

Unmodified V3 screening baseline: 49/400. Screen seeds 6310001–6310400, shared across candidates. Highest screening count selected, ties control,A,C,B. Small differences between candidates are not proof of a universally better algorithm. A changes both data and training policy; B changes capacity and trainable weights; C also changes nominal learning rate. This is not a single-factor ablation.

## Runtime validation

Native/model tests: zero-residual identity, frozen base, normalized symmetry-aware update, D4 invariance, serialization and TC cancellation. Browser parity checks compare actual selected model's direction, depth and visited nodes against C++; see validation.json. Actual worker tests include V4 and chained load failures. Browser input tests cover takeover, stopping, restart and fallback.

The same board-only leaf cache optimization is applied to both V3 and V4 JS score searches, preserving all fixed-node decisions. Actual JS search was also tested at the unchanged 160 ms soft deadline on shared sampled positions:

| Policy | Positions | Mean ms | p95 ms | Changed vs unlimited |
|---|---:|---:|---:|---:|
| V3 | 98 | 13.70 | 31.76 | 0 |
| V4 | 98 | 12.21 | 30.78 | 0 |

This runs actual browser search code in Node on the evaluation machine; it is not an iPhone success-rate measurement. The original game's bonus-preview posterior is reconstructed and not proven identical to the official app. No claim of comparison with external SOTA is made.

Frozen base SHA256: `6c4a8d47d760d55a3f2ab4e7fbd9feb6345efa13fa9c870567a9d86e0892356b`. Residual SHA256: `None`. All candidate weights are archived with training logs, data-source provenance and individual-game results in training/v4. CPU-budget stopping means exact retraining episode counts vary by machine; saved weights define this evaluated experiment.
