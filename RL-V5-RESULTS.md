# V5 corner shaping results

Screen-selected release: **strong**, coefficient **32768**. Default promotion: **False**. Best shaping candidate frozen before holdout: **strong**.

## New independent test

5000 shared seeds 7410001–7415000 per policy. Same reconstructed game, 3-ply expectimax, 24000 nodes; offline timing unlimited. First 6144 or actual terminal, no truncations. Selection used only the separate 400-game screen. No holdout-driven retraining or candidate switching.

| Policy | 6144 count | Success rate | Wilson 95% CI | Reach 3072 | 6144 given 3072 | Maximum in corner during late play |
|---|---:|---:|---:|---:|---:|---:|
| V4 baseline | 721/5000 | 14.42% | 13.47%–15.42% | 65.56% | 22.00% | 27.97% |
| V4 + more training | 745/5000 | 14.90% | 13.94%–15.91% | 65.56% | 22.73% | 32.27% |
| Corner shaping (strong) | 727/5000 | 14.54% | 13.59%–15.54% | 65.56% | 22.18% | 72.91% |

Corner occupancy is pooled across decisions with maximum 3072, weighted by move count, not by game. It is a diagnostic, not the selection metric. Higher corner occupancy alone is not evidence of better play.

| Paired comparison (second minus first) | Difference, pp | Paired 95% interval, pp | Exact McNemar p |
|---|---:|---:|---:|
| v4_vs_control | 0.48 | -0.78 to 1.74 | 0.47274 |
| v4_vs_shaping | 0.12 | -1.13 to 1.37 | 0.87536 |
| control_vs_shaping | -0.36 | -1.63 to 0.91 | 0.59846 |

Primary promotion comparison: **v4_vs_shaping**. Other comparisons are descriptive, not multiplicity-adjusted. Same seed does not imply identical later random outcomes after policies diverge.

## Equal-CPU training and screening

| Arm | Coefficient | Training episodes | Process CPU seconds | 6144 / 400 |
|---|---:|---:|---:|---:|
| control | 0 | 237799 | 450.002 | 64 |
| weak | 2048 | 235727 | 450.001 | 70 |
| medium | 8192 | 244594 | 450 | 63 |
| strong | 32768 | 251250 | 450 | 71 |

All arms start with V4 raw weights and the same training seed, old late-state pool, 20% recent replay, two-ply training policy, five-step TD(lambda=.5), and learning-rate schedule. The corner coefficient is the experimental variable. Each trainer gets 450 process CPU seconds plus the remainder of its final episode. Counts differ because computation and trajectories differ. Initial loading is excluded. The first two stage tables remain unchanged.

## Meaning of the intervention

Potential is 1 when a maximum tile of at least 3072 is in any corner of an afterstate. Terminal potential is zero. Training uses r' = r + c(Phi(next)-Phi(current)), including the final subtraction. Raw model output U predicts shaped return; native and browser search restore the original-score estimate with U + c Phi and use original game rewards. This is not a repeating per-move corner bonus.

Raw tables were initialized identically, making the initial corrected estimate V4 + c Phi. Thus this experiment includes a deliberate corner-optimistic value initialization. Exact compensation at initialization would largely cancel the shaping transformation. The exact shaping identity preserves the original score objective, not necessarily the probability of 6144; finite N-tuple learning and approximate search provide no guarantee of improvement. One training seed per arm and one budget do not establish general superiority or inferiority of corner strategies.

## Runtime checks

Native tests cover terminal subtraction, return telescoping, five-step lambda equivalence, corrected inference, stage isolation and symmetry. Browser tests cover the same correction, invalid configuration, and exact native direction/depth/node parity on sampled positions. Worker tests cover all five learned versions and chained loading failures; input tests cover fallback and continuing play. See validation.json and worker-results.json.

| Policy | Sampled positions | Mean ms | p95 ms | Changed from unlimited at 160 ms |
|---|---:|---:|---:|---:|
| v4 | 98 | 19.79 | 52.26 | 0 |
| v5 | 98 | 21.79 | 44.81 | 0 |

Actual browser search code runs in Node on this machine; not a phone-specific success estimate. Reconstructed bonus-preview rules are not independently proven identical to the official app. Frozen selected SHA256: `8d65f093ee47450f7d42140d032a6fcd969ec374f2584e7ecb2eb716d8f748cb`. Full weights, logs, protocol, and individual-game records are retained under training/v5.
