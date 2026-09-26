from pathlib import Path
import json,hashlib,shutil
v=Path('training/v4');d=Path('dist/models');s=json.loads((v/'selection.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(v/'selected.ntd')==s['scoreSha256'];shutil.copyfile(v/'selected.ntd',d/'ntuple-v4.bin');kind='residual-five' if s['residualSha256'] else 'score'
if s['residualSha256']:
 assert sha(v/'selected.five')==s['residualSha256'];shutil.copyfile(v/'selected.five',d/'residual-v4.bin')
(d/'v4-policy.json').write_text(json.dumps({'kind':kind,'candidate':s['winner'],'scoreSha256':s['scoreSha256'],'residualSha256':s['residualSha256']},indent=2)+'\n')
training={}
for n in ['control','a','b','c']:
 rows=[json.loads(l) for l in (v/f'run-{n}/progress.jsonl').read_text().splitlines()];training[n]=rows[-1]
meta={'version':4,'kind':kind,'selection':s,'training':training,'search':{'maxDepth':3,'maxNodes':24000,'browserSoftBudgetMs':160},'bytes':(d/'ntuple-v4.bin').stat().st_size+((d/'residual-v4.bin').stat().st_size if kind=='residual-five' else 0),'protocol':json.loads((v/'protocol.json').read_text())};(d/'ntuple-v4.json').write_text(json.dumps(meta,indent=2)+'\n');print({'winner':s['winner'],'kind':kind,'bytes':meta['bytes']})
