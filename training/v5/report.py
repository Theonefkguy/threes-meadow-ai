from pathlib import Path
import json,re,hashlib
v=Path('training/v5');e=json.loads((v/'evaluation.json').read_text());s=e['selection'];p=json.loads((v/'protocol.json').read_text());timing=json.loads((v/'timing.json').read_text());validation=json.loads((v/'validation.json').read_text());e['browserGate']=timing['compatible'];e['promote']=e['statisticalPromotionGate'] and e['browserGate'];(v/'evaluation.json').write_text(json.dumps(e,indent=2)+'\n')
pct=lambda x:f'{100*x:.2f}%';labels={'v4':'V4 baseline','control':'V4 + more training','shaping':f'Corner shaping ({s["bestShaping"]})'}
text=f'''# V5 corner shaping results

Screen-selected release: **{s['winner']}**, coefficient **{s['coefficient']}**. Default promotion: **{e['promote']}**. Best shaping candidate frozen before holdout: **{s['bestShaping']}**.

## New independent test

5000 shared seeds 7410001–7415000 per policy. Same reconstructed game, 3-ply expectimax, 24000 nodes; offline timing unlimited. First 6144 or actual terminal, no truncations. Selection used only the separate 400-game screen. No holdout-driven retraining or candidate switching.

| Policy | 6144 count | Success rate | Wilson 95% CI | Reach 3072 | 6144 given 3072 | Maximum in corner during late play |
|---|---:|---:|---:|---:|---:|---:|
'''
for name,a in e['policies'].items():text+=f"| {labels[name]} | {a['wins6144']}/5000 | {pct(a['rate6144'])} | {pct(a['wilson95'][0])}–{pct(a['wilson95'][1])} | {pct(a['rate3072'])} | {pct(a['rate6144Given3072'])} | {pct(a['cornerRateLate'])} |\n"
text+='\nCorner occupancy is pooled across decisions with maximum 3072, weighted by move count, not by game. It is a diagnostic, not the selection metric. Higher corner occupancy alone is not evidence of better play.\n\n| Paired comparison (second minus first) | Difference, pp | Paired 95% interval, pp | Exact McNemar p |\n|---|---:|---:|---:|\n'
for name,a in e['comparisons'].items():text+=f"| {name} | {100*a['difference']:.2f} | {100*a['difference95'][0]:.2f} to {100*a['difference95'][1]:.2f} | {a['exactMcNemarP']:.5f} |\n"
text+=f"\nPrimary promotion comparison: **{e['primaryComparison']}**. Other comparisons are descriptive, not multiplicity-adjusted. Same seed does not imply identical later random outcomes after policies diverge.\n\n## Equal-CPU training and screening\n\n| Arm | Coefficient | Training episodes | Process CPU seconds | 6144 / 400 |\n|---|---:|---:|---:|---:|\n"
for n,c in p['coefficients'].items():
 a=json.loads((v/f'run-{n}/progress.jsonl').read_text().splitlines()[-1]);text+=f"| {n} | {c} | {a['episodes']} | {a['cpuSeconds']} | {s['screening'][n]['wins6144']} |\n"
text+='''
All arms start with V4 raw weights and the same training seed, old late-state pool, 20% recent replay, two-ply training policy, five-step TD(lambda=.5), and learning-rate schedule. The corner coefficient is the experimental variable. Each trainer gets 450 process CPU seconds plus the remainder of its final episode. Counts differ because computation and trajectories differ. Initial loading is excluded. The first two stage tables remain unchanged.

## Meaning of the intervention

Potential is 1 when a maximum tile of at least 3072 is in any corner of an afterstate. Terminal potential is zero. Training uses r' = r + c(Phi(next)-Phi(current)), including the final subtraction. Raw model output U predicts shaped return; native and browser search restore the original-score estimate with U + c Phi and use original game rewards. This is not a repeating per-move corner bonus.

Raw tables were initialized identically, making the initial corrected estimate V4 + c Phi. Thus this experiment includes a deliberate corner-optimistic value initialization. Exact compensation at initialization would largely cancel the shaping transformation. The exact shaping identity preserves the original score objective, not necessarily the probability of 6144; finite N-tuple learning and approximate search provide no guarantee of improvement. One training seed per arm and one budget do not establish general superiority or inferiority of corner strategies.

## Runtime checks

Native tests cover terminal subtraction, return telescoping, five-step lambda equivalence, corrected inference, stage isolation and symmetry. Browser tests cover the same correction, invalid configuration, and exact native direction/depth/node parity on sampled positions. Worker tests cover all five learned versions and chained loading failures; input tests cover fallback and continuing play. See validation.json and worker-results.json.

| Policy | Sampled positions | Mean ms | p95 ms | Changed from unlimited at 160 ms |
|---|---:|---:|---:|---:|
'''
for n in ['v4','v5']:
 a=timing[n];text+=f"| {n} | {a['positions']} | {a['meanMs']:.2f} | {a['p95Ms']:.2f} | {a['changedFromUnlimited']} |\n"
text+=f"\nActual browser search code runs in Node on this machine; not a phone-specific success estimate. Reconstructed bonus-preview rules are not independently proven identical to the official app. Frozen selected SHA256: `{s['hashes'][s['winner']]}`. Full weights, logs, protocol, and individual-game records are retained under training/v5.\n"
Path('RL-V5-RESULTS.md').write_text(text);Path('RL-V5.md').write_text((v/'README.md').read_text())
html=Path('dist/index.html').read_text().replace('?v=10','?v=11');html=re.sub(r'\s*<option value="rl-v5"[^>]*>.*?</option>','',html)
label='强化学习 · 第五版' if e['promote'] else '强化学习 · 第五版（实验）';option='<option value="rl-v5"'+(' selected' if e['promote'] else '')+'>'+label+'</option>'
if e['promote']:html=html.replace('<option value="rl-v4" selected>','<option value="rl-v4">')
html=html.replace('<option value="rl-v4"',option+'\n        <option value="rl-v4"',1)
note='第五版首次使用需加载约 6 MB，AI 在本机运行。保留旧版，可切换对比。' if e['promote'] else '第五版为角落实验，暂未证实更高成功率。模型约 6 MB，AI 在本机运行。'
html=re.sub(r'(<p class="ai-note">).*?(</p>)',lambda m:m[1]+note+m[2],html);Path('dist/index.html').write_text(html)
m=Path('dist/models/ntuple-v5.json');data=json.loads(m.read_text());data.update(evaluation=e,timing=timing,validation=validation);m.write_text(json.dumps(data,indent=2)+'\n');print(text)
