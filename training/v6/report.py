from pathlib import Path
import json,re
v=Path('training/v6');e=json.loads((v/'evaluation.json').read_text());s=e['selection'];timing=json.loads((v/'timing.json').read_text());validation=json.loads((v/'validation.json').read_text());e['browserGate']=timing['compatible'];e['promote']=e['statisticalPromotionGate'] and e['browserGate'];(v/'evaluation.json').write_text(json.dumps(e,indent=2)+'\n')
pct=lambda x:f'{100*x:.2f}%';labels={'v4':'V4 baseline','control':'Ordinary early-stage continued training','selected':s['selected']}
text=f'''# V6 early-stage results

Screen-selected arm: **{s['selected']}**. Policy: **{s['kind']}**. Default promotion: **{e['promote']}**.

5000 shared new seeds8410001–8415000 per policy, from ordinary openings until first6144 or actual death. All later-stage weights remain exactly V4. Native3-ply search,24000 nodes, no wall-time deadline. No truncated games. The selected arm was frozen before inspecting the holdout.

| Policy | 1536 count | Reach1536 | Reach3072 | Reach6144 | 6144 count | 6144 given1536 |
|---|---:|---:|---:|---:|---:|---:|
'''
for name,a in e['policies'].items():text+=f"| {labels[name]} | {a['counts']['1536']}/5000 | {pct(a['rates']['1536'])} | {pct(a['rates']['3072'])} | {pct(a['rates']['6144'])} | {a['counts']['6144']}/5000 | {pct(a['rate6144Given1536'])} |\n"
text+='\n| Paired comparison (second minus first) | Target | Difference, pp | Paired95% interval, pp | Exact McNemar p |\n|---|---:|---:|---:|---:|\n'
for name,group in e['comparisons'].items():
 for target,a in group.items():text+=f"| {name} | {target} | {100*a['difference']:.2f} | {100*a['difference95'][0]:.2f} to {100*a['difference95'][1]:.2f} | {a['exactMcNemarP']:.5f} |\n"
text+=f"\nPrimary comparison **{e['primaryComparison']}**. Early improvement gate: **{e['earlyGate']}**. End-to-end6144 gate: **{e['lateGate']}**. The latter permits a95% lower bound above−1pp, while requiring a nonnegative point difference; this does not prove zero harm. Secondary comparisons are descriptive and not multiplicity-adjusted. Per-policy Wilson intervals and failure maxima are in evaluation.json.\n\n## Equal-CPU training and independent screening\n\n| Arm | Episodes | CPU seconds | Warmstart updates |1536/400 |3072/400 |6144/400 |\n|---|---:|---:|---:|---:|---:|---:|\n"
for arm in ['control','hard','probability']:
 a=json.loads((v/f'run-{arm}/progress.jsonl').read_text().splitlines()[-1]);r=s['screen'][arm];text+=f"| {arm} | {a['episodes']} | {a['cpuSeconds']} | {a['warmUpdates']} | {r['1536']} | {r['3072']} | {r['6144']} |\n"
collection=[json.loads(l) for l in (v/'games.jsonl').read_text().splitlines()];assessment=[json.loads(l) for l in (v/'assessment.jsonl').read_text().splitlines()];hard=json.loads((v/'hard-indices.json').read_text())
text+=f"\nCollected2000 fresh V4 trajectories; {sum(not x['success1536'] for x in collection)} failed before1536. Assessed{len(assessment)} candidate snapshots, selected{len(hard)} with mixed continuation outcomes. No holdout states entered training.\n\nControl and hard arms differ only in resume-pool distribution. Probability additionally changes the objective and spends90 of450 CPU seconds on supervised initialization; this is not a pure reward-only comparison. Negative results would concern this implementation/budget, not all goal-directed reinforcement learning. Exact archived weights, one training seed per arm.\n\n## Runtime checks\n\n"
text+=json.dumps(validation,ensure_ascii=False,indent=2)+'\n\n| Policy | Positions | Mean ms |p95 ms |Changed from unlimited at160ms |\n|---|---:|---:|---:|---:|\n'
for name in ['v4','v6']:
 a=timing[name];text+=f"| {name} | {a['positions']} | {a['meanMs']:.2f} | {a['p95Ms']:.2f} | {a['changedFromUnlimited']} |\n"
text+=f"\nActual browser search code in Node on this machine, not a phone success estimate. Reconstructed bonus-preview rules are not independently proven identical to the official app. Frozen selected SHA256 `{s['sha256']}`. See training/v6/README.md for objective boundaries, sampling, reproduction, and limitations.\n"
Path('RL-V6-RESULTS.md').write_text(text);Path('RL-V6.md').write_text((v/'README.md').read_text())
html=Path('dist/index.html').read_text();html=re.sub(r'\s*<option value="rl-v6"[^>]*>.*?</option>','',html)
label='强化学习 · 第六版' if e['promote'] else '强化学习 · 第六版（实验）';option='<option value="rl-v6"'+(' selected' if e['promote'] else '')+'>'+label+'</option>'
html=html.replace('<option value="rl-v4" selected>','<option value="rl-v4">') if e['promote'] else html
html=html.replace('<option value="rl-v5"',option+'\n        <option value="rl-v5"',1)
note='第六版着重提升前期成功率。模型约 6 MB，AI 在本机运行。' if e['promote'] else '第六版为前期强化学习实验，默认保留第四版。模型约 6 MB，AI 在本机运行。'
html=re.sub(r'(<p class="ai-note">).*?(</p>)',lambda m:m[1]+note+m[2],html);Path('dist/index.html').write_text(html)
Path('dist/models/ntuple-v6.json').write_text(json.dumps({'evaluation':e,'timing':timing,'validation':validation},indent=2)+'\n');print(json.dumps({'promote':e['promote'],'selected':s['selected'],'rates':{n:p['rates'] for n,p in e['policies'].items()}},indent=2))
