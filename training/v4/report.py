from pathlib import Path
import json
v=Path('training/v4');e=json.loads((v/'evaluation.json').read_text());s=e['selection'];val=json.loads((v/'validation.json').read_text());timing=json.loads((v/'timing.json').read_text());e['browserGate']=timing['compatible'];e['promote']=e['statisticalPromotionGate'] and e['browserGate'];(v/'evaluation.json').write_text(json.dumps(e,indent=2)+'\n');a,b=e['v3'],e['v4'];pair=e['paired'];pct=lambda x:f'{x*100:.2f}%'
text=f'''# Threes V4 research results

Selected candidate: **{s['winner']}**. Promoted as default: **{e['promote']}**.

## Independent 5000-game test

New seeds 6410001–6415000; same reconstructed engine, 3-ply expectimax and 24000-node budget; no wall-clock limit offline. First 6144 or actual terminal, no truncations. Selection/frozen hashes preceded inspection of this test. These are fresh estimates, not the previous test's results.

| Metric | V3 | V4 |
|---|---:|---:|
| Reached 6144 | {a['wins6144']}/5000 | {b['wins6144']}/5000 |
| 6144 rate | {pct(a['rate6144'])} | {pct(b['rate6144'])} |
| 6144 Wilson 95% CI | {pct(a['wilson95'][0])}–{pct(a['wilson95'][1])} | {pct(b['wilson95'][0])}–{pct(b['wilson95'][1])} |
| Reached 3072 | {pct(a['rate3072'])} | {pct(b['rate3072'])} |
| 6144 given 3072 | {pct(a['rate6144Given3072'])} | {pct(b['rate6144Given3072'])} |

Paired difference {100*pair['difference']:.2f} percentage points, approximate paired 95% CI [{100*pair['difference95'][0]:.2f}, {100*pair['difference95'][1]:.2f}] pp. Exact two-sided McNemar p={pair['exactMcNemarP']}. V3-only successes {pair['v3Only']}; V4-only {pair['v4Only']}. A shared seed does not guarantee identical subsequent random outcomes after policies diverge.

## Compute-matched screening

Each arm received approximately 450 process CPU seconds, excluding initial loading; final episode allowed to finish. Shared new-data collection (1000 V3 games yielding 4854 states from 680 source games) is additional. Training times include policy search and learning overhead; episode counts intentionally differ.

| Arm | Change | Training episodes | CPU seconds | 6144 / 400 screening games |
|---|---|---:|---:|---:|
'''
labels={'control':'Continue V3 on historical late pool, 2-ply','a':'Fresh pool + 5% 3-ply episodes','b':'A data/policy + frozen V3, five-cell residual','c':'A data/policy + Temporal Coherence'}
for n in ['control','a','b','c']:
 t=val['training'][n]['summary'];text+=f"| {n} | {labels[n]} | {t['episodes']} | {t['cpuSeconds']} | {s['screening'][n]['wins6144']}/400 |\n"
text+=f'''
Unmodified V3 screening baseline: {s['screening']['v3']['wins6144']}/400. Screen seeds 6310001–6310400, shared across candidates. Highest screening count selected, ties control,A,C,B. Small differences between candidates are not proof of a universally better algorithm. A changes both data and training policy; B changes capacity and trainable weights; C also changes nominal learning rate. This is not a single-factor ablation.

## Runtime validation

Native/model tests: zero-residual identity, frozen base, normalized symmetry-aware update, D4 invariance, serialization and TC cancellation. Browser parity checks compare actual selected model's direction, depth and visited nodes against C++; see validation.json. Actual worker tests include V4 and chained load failures. Browser input tests cover takeover, stopping, restart and fallback.

The same board-only leaf cache optimization is applied to both V3 and V4 JS score searches, preserving all fixed-node decisions. Actual JS search was also tested at the unchanged 160 ms soft deadline on shared sampled positions:

| Policy | Positions | Mean ms | p95 ms | Changed vs unlimited |
|---|---:|---:|---:|---:|
| V3 | {timing['v3']['positions']} | {timing['v3']['meanMs']:.2f} | {timing['v3']['p95Ms']:.2f} | {timing['v3']['changedFromUnlimited']} |
| V4 | {timing['v4']['positions']} | {timing['v4']['meanMs']:.2f} | {timing['v4']['p95Ms']:.2f} | {timing['v4']['changedFromUnlimited']} |

This runs actual browser search code in Node on the evaluation machine; it is not an iPhone success-rate measurement. The original game's bonus-preview posterior is reconstructed and not proven identical to the official app. No claim of comparison with external SOTA is made.

Frozen base SHA256: `{s['scoreSha256']}`. Residual SHA256: `{s['residualSha256']}`. All candidate weights are archived with training logs, data-source provenance and individual-game results in training/v4. CPU-budget stopping means exact retraining episode counts vary by machine; saved weights define this evaluated experiment.
'''
Path('RL-V4-RESULTS.md').write_text(text);Path('RL-V4.md').write_text((v/'README.md').read_text());print(text)
# Update product only after both gates.
p=Path('dist/index.html');html=p.read_text().replace('?v=9','?v=10');label='强化学习 · 第四版' if e['promote'] else '强化学习 · 第四版（实验）';option=f'<option value="rl-v4"'+(' selected' if e['promote'] else '')+f'>{label}</option>'
if e['promote']:html=html.replace('<option value="rl-v3" selected>','<option value="rl-v3">')
html=html.replace('<option value="rl-v3"',option+'\n        <option value="rl-v3"',1)
size=14 if s['residualSha256'] else 6
html=html.replace('首次使用需加载约 6 MB，AI 在本机运行。第三版针对合出 6144 训练，可切换旧版对比。',f'第四版首次使用需加载约 {size} MB，AI 在本机运行。保留旧版，可切换对比。')
p.write_text(html)
p=Path('dist/models/ntuple-v4.json');m=json.loads(p.read_text());m['evaluation']=e;m['timing']=timing;p.write_text(json.dumps(m,indent=2)+'\n')
