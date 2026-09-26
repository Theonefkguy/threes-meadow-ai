from pathlib import Path
import json,re
v=Path('training/v7');e=json.loads((v/'evaluation.json').read_text());s=e['selection'];timing=json.loads((v/'timing.json').read_text());validation=json.loads((v/'validation.json').read_text());e['browserGate']=timing['compatible'];e['promote']=e['statisticalPromotionGate'] and e['browserGate'];(v/'evaluation.json').write_text(json.dumps(e,indent=2)+'\n');pct=lambda x:f'{100*x:.2f}%'
labels={'baseline':'V6 ordinary continued-training baseline','selected':'V7 '+s['selected'],'v4':'V4 website default'}
text=f'''# V7 early-stage experiment results

Selected arm: **{s['selected']}**. Default promotion: **{e['promote']}**.

## New5000-game test

Paired seeds9410001–9415000, normal openings to first6144 or actual death. No truncated games. Same3-ply/24000 nodes, unlimited offline time. Model selected and frozen before holdout inspection.

| Policy |1536 count |Reach1536 |Reach3072 |Reach6144 |6144 count |6144 given1536 |
|---|---:|---:|---:|---:|---:|---:|
'''
for name,a in e['policies'].items():text+=f"| {labels[name]} | {a['counts']['1536']}/5000 | {pct(a['rates']['1536'])} | {pct(a['rates']['3072'])} | {pct(a['rates']['6144'])} | {a['counts']['6144']}/5000 | {pct(a['rate6144Given1536'])} |\n"
text+='\n| Comparison (selected minus baseline) |Target |Difference, pp |Paired95% interval, pp |Exact McNemar p |\n|---|---:|---:|---:|---:|\n'
for name,group in e['comparisons'].items():
 for target,a in group.items():text+=f"| {name} | {target} | {100*a['difference']:.2f} | {100*a['difference95'][0]:.2f} to {100*a['difference95'][1]:.2f} | {a['exactMcNemarP']:.5f} |\n"
text+='\nPromotion gates: '+json.dumps(e['gates'])+'. The primary goal is improved1536 versus the starting model. Promotion additionally checks V4 and a−1pp6144 uncertainty margin with nonnegative6144 point differences. Other reported comparisons are descriptive, not multiplicity adjusted. See evaluation.json for all rate intervals.\n\n## Equal training budget and separate600-game screen\n\n| Arm |Hard-start ratio |Teacher replay |Episodes |CPU seconds |1536/600 |3072/600 |6144/600 |\n|---|---:|---|---:|---:|---:|---:|---:|\n'
protocol=json.loads((v/'protocol.json').read_text())
for arm,config in protocol['arms'].items():
 a=json.loads((v/f'run-{arm}/progress.jsonl').read_text().splitlines()[-1]);r=s['screen'][arm];text+=f"| {arm} | {pct(config['hardRatio'])} | {config['teacher']} | {a['episodes']} | {a['cpuSeconds']} | {r['1536']} | {r['3072']} | {r['6144']} |\n"
a=json.loads((v/'data-summary.json').read_text());text+='\nFresh data: '+json.dumps({k:x for k,x in a.items() if k!='sha256'})+'.\n'
text+='\nAll arms preserve later V4 tables, use lower learning rates and common frozen-value anchors. Ratios isolate replay-distribution effects; r10teacher versus r10 isolates added value-target distillation within the fixed training budget. Teacher-generation compute is additional and outside that budget. The stronger teacher predicts score, not necessarily1536 success. A negative result does not establish that all curriculum learning or distillation is ineffective.\n\n## Runtime validation\n\n'
text+='```json\n'+json.dumps(validation,ensure_ascii=False,indent=2)+'\n```\n\n| Policy |Positions |Mean ms |p95 ms |Changed from unlimited at160ms |\n|---|---:|---:|---:|---:|\n'
for name in ['v4','v7']:
 a=timing[name];text+=f"| {name} | {a['positions']} | {a['meanMs']:.2f} | {a['p95Ms']:.2f} | {a['changedFromUnlimited']} |\n"
text+=f"\nBrowser search code measured in Node on this machine, not a phone-specific success rate. Reconstructed game rules. Single training seed and budget. Selected model SHA256: `{s['sha256']}`. Code, exact weights, source data and individual-game results are retained under training/v7.\n"
Path('RL-V7-RESULTS.md').write_text(text);Path('RL-V7.md').write_text((v/'README.md').read_text())
p=Path('dist/index.html');html=p.read_text();html=re.sub(r'\s*<option value="rl-v7"[^>]*>.*?</option>','',html);option='<option value="rl-v7"'+(' selected' if e['promote'] else '')+'>'+('强化学习 · 第七版' if e['promote'] else '强化学习 · 第七版（实验）')+'</option>'
if e['promote']:html=html.replace('<option value="rl-v4" selected>','<option value="rl-v4">')
html=html.replace('<option value="rl-v6"',option+'\n        <option value="rl-v6"',1);note='第七版着重提升前期成功率。模型约 6 MB，AI 在本机运行。' if e['promote'] else '第七版为前期训练实验，默认保留第四版。模型约 6 MB，AI 在本机运行。';html=re.sub(r'(<p class="ai-note">).*?(</p>)',lambda m:m[1]+note+m[2],html);p.write_text(html)
Path('dist/models/ntuple-v7.json').write_text(json.dumps({'evaluation':e,'timing':timing,'validation':validation},indent=2)+'\n');print(json.dumps({'selected':s['selected'],'promote':e['promote'],'rates':{n:p['rates'] for n,p in e['policies'].items()}},indent=2))
