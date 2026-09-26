# Threes V3 — 6144 experiments

Selected candidate: **C**. Promoted as default: **True**.

## Independent normal-opening evaluation

Same 5000 seeds, 3910001–3915000. Maximum search depth 3, node budget 24000, no wall-clock cutoff. Stop at first 6144 or terminal. No truncated games.

| Policy | Reached 3072 | Reached 6144 | 6144 95% Wilson CI | 6144 given 3072 |
|---|---:|---:|---:|---:|
| V2 | 3181/5000 (63.62%) | 403/5000 (8.06%) | 7.34%–8.85% | 12.67% |
| V3 | 3249/5000 (64.98%) | 639/5000 (12.78%) | 11.88%–13.73% | 19.67% |

Paired difference: 4.72 percentage points; approximate paired 95% CI [3.54, 5.90] pp. V2-only wins 342, V3-only wins 578. Exact two-sided McNemar p=6.692852317905942e-15.

The same seeds do not force the same tile positions once policies diverge. This is a matched-seed experiment, not identical trajectories.

## Candidate screening (not final estimates)

200 fresh normal openings per policy, seeds 2910001–2910200. Selection uses these results only.

| Candidate | Additional training | 6144 successes |
|---|---|---:|
| V2 baseline | none | 13/200 |
| A | 200000 late score episodes; earlier stages frozen | 22/200 |
| B | A + 100000 probability-head episodes | 20/200 |
| C | A + 100000 earlier-stage score episodes | 23/200 |

Frozen score SHA256: `adfbad1882b42307fc39810d9f88bd59ed8005829570bacccaf174bcfc14f946`. Goal SHA256: `None`. Weights frozen before final holdout results were read. No tuning based on the final test.

## Fixed late-position diagnostic

Collect playable 3072 positions from 200 separate V2 normal games, seeds 4910001–4910200; five independent continuations per position, with remaining hidden deck order shuffled conditional on public counts. Policies see only board, hint and public counts. These are repeated continuations of shared positions, **not** independent normal-opening games.

| Policy | Successes | Rate |
|---|---:|---:|
| V2 | 92/645 | 14.26% |
| V3 | 115/645 | 17.83% |

## Interpretation limits

All numbers use our reconstructed Threes engine. In particular, the bonus hint posterior has not been proven identical to the official app. Browser play has a 160 ms soft cutoff; offline success rates are not phone-specific rates. Normal-opening conditional success compares different reached positions; the fixed late-position diagnostic isolates shared starts. Candidate training budgets and objectives differ, so the experiment is not a compute-matched causal ablation.

See `training/v3/protocol.json`, `selection.json`, `evaluation.json`, `validation.json`, per-game JSONL and trainer sources for reproduction. Raw model checkpoints for non-selected candidates are preserved compressed; selected weights are also in `dist/models/` when deployed.
