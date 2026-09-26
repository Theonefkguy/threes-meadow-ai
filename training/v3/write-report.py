import json,hashlib,gzip
from pathlib import Path
p=Path('training/v3');e=json.loads((p/'evaluation.json').read_text());sel=e['selection'];v2=e['v2'];v3=e['v3'];paired=e['paired'];pct=lambda x:f'{100*x:.2f}%'
# Paired independent continuation seeds, clustered by board when interpreting.
late={}
for name in ['v2','v3']:
 rows=[json.loads(l) for l in (p/f'late-{name}.jsonl').read_text().splitlines()];assert not any(r['truncated'] for r in rows);late[name]={'states':len(set(r['state'] for r in rows)),'rollouts':len(rows),'wins':sum(r['success'] for r in rows),'rate':sum(r['success'] for r in rows)/len(rows)}
e['lateDiagnostic']=late;(p/'evaluation.json').write_text(json.dumps(e,indent=2)+'\n')
text=f'''# Threes V3 — 6144 experiments

Selected candidate: **{sel['winner'].upper()}**. Promoted as default: **{e['promote']}**.

## Independent normal-opening evaluation

Same 5000 seeds, 3910001–3915000. Maximum search depth 3, node budget 24000, no wall-clock cutoff. Stop at first 6144 or terminal. No truncated games.

| Policy | Reached 3072 | Reached 6144 | 6144 95% Wilson CI | 6144 given 3072 |
|---|---:|---:|---:|---:|
| V2 | {v2['reached3072']}/5000 ({pct(v2['rate3072'])}) | {v2['reached6144']}/5000 ({pct(v2['rate6144'])}) | {pct(v2['wilson95'][0])}–{pct(v2['wilson95'][1])} | {pct(v2['rate6144Given3072'])} |
| V3 | {v3['reached3072']}/5000 ({pct(v3['rate3072'])}) | {v3['reached6144']}/5000 ({pct(v3['rate6144'])}) | {pct(v3['wilson95'][0])}–{pct(v3['wilson95'][1])} | {pct(v3['rate6144Given3072'])} |

Paired difference: {100*paired['difference']:.2f} percentage points; approximate paired 95% CI [{100*paired['difference95'][0]:.2f}, {100*paired['difference95'][1]:.2f}] pp. V2-only wins {paired['v2Only']}, V3-only wins {paired['v3Only']}. Exact two-sided McNemar p={paired['mcnemarExactP']}.

The same seeds do not force the same tile positions once policies diverge. This is a matched-seed experiment, not identical trajectories.

## Candidate screening (not final estimates)

200 fresh normal openings per policy, seeds 2910001–2910200. Selection uses these results only.

| Candidate | Additional training | 6144 successes |
|---|---|---:|
| V2 baseline | none | {sel['screening']['v2']['wins']}/200 |
| A | 200000 late score episodes; earlier stages frozen | {sel['screening']['a']['wins']}/200 |
| B | A + 100000 probability-head episodes | {sel['screening']['b']['wins']}/200 |
| C | A + 100000 earlier-stage score episodes | {sel['screening']['c']['wins']}/200 |

Frozen score SHA256: `{sel['scoreSha256']}`. Goal SHA256: `{sel['goalSha256']}`. Weights frozen before final holdout results were read. No tuning based on the final test.

## Fixed late-position diagnostic

Collect playable 3072 positions from 200 separate V2 normal games, seeds 4910001–4910200; five independent continuations per position, with remaining hidden deck order shuffled conditional on public counts. Policies see only board, hint and public counts. These are repeated continuations of shared positions, **not** independent normal-opening games.

| Policy | Successes | Rate |
|---|---:|---:|
| V2 | {late['v2']['wins']}/{late['v2']['rollouts']} | {pct(late['v2']['rate'])} |
| V3 | {late['v3']['wins']}/{late['v3']['rollouts']} | {pct(late['v3']['rate'])} |

## Interpretation limits

All numbers use our reconstructed Threes engine. In particular, the bonus hint posterior has not been proven identical to the official app. Browser play has a 160 ms soft cutoff; offline success rates are not phone-specific rates. Normal-opening conditional success compares different reached positions; the fixed late-position diagnostic isolates shared starts. Candidate training budgets and objectives differ, so the experiment is not a compute-matched causal ablation.

See `training/v3/protocol.json`, `selection.json`, `evaluation.json`, `validation.json`, per-game JSONL and trainer sources for reproduction. Raw model checkpoints for non-selected candidates are preserved compressed; selected weights are also in `dist/models/` when deployed.
'''
Path('RL-V3-RESULTS.md').write_text(text)
Path('RL-V3.md').write_text((p/'README.md').read_text())
for source,name in [(p/'a.ntd','candidate-a.ntd.gz'),(p/'run-b/checkpoint.goal','candidate-b.goal.gz'),(p/'run-c/checkpoint.ntd','candidate-c.ntd.gz')]:
 with gzip.GzipFile(str(p/name),'wb',mtime=0) as f:f.write(source.read_bytes())
print(text)
