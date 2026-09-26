import json,hashlib,shutil
from pathlib import Path
v=Path('training/v3');d=Path('dist');sel=json.loads((v/'selection.json').read_text());score=v/'selected.ntd';assert hashlib.sha256(score.read_bytes()).hexdigest()==sel['scoreSha256'];shutil.copyfile(score,d/'models/ntuple-v3.bin')
kind='late-goal' if sel['goalSha256'] else 'score'
if kind!='score':raise RuntimeError('This browser build ships the selected score policy; the probability-head candidate is offline only.')
if kind=='late-goal':
 goal=v/'selected.goal';assert hashlib.sha256(goal.read_bytes()).hexdigest()==sel['goalSha256'];shutil.copyfile(goal,d/'models/goal-v3.bin')
else:
 # Only the shipped score policy needs to be in the product. Probability-head
 # experiments remain reproducible in training/v3, not unused browser branches.
 (d/'v3-policy.js').write_text("import {loadModel} from './rl-model.js?v=9';\nexport async function loadV3Policy(){return {score:await loadModel('v3')};}\n")
 p=d/'rl-search.js';p.write_text(p.read_text().replace(',terminalBonus=0}={}', '}={}').replace('return Math.max(...b)>=14?terminalBonus:0;', 'return 0;'))
 p=d/'ai-worker.js';p.write_text(p.read_text().replace('{...options,terminalBonus:model.terminalBonus}', 'options'))
config={'kind':kind,'scoreVersion':'v3','goalBonus':200000 if kind=='late-goal' else 0,'candidate':sel['winner'],'scoreSha256':sel['scoreSha256'],'goalSha256':sel['goalSha256']};(d/'models/v3-policy.json').write_text(json.dumps(config,indent=2)+'\n')
meta={'name':'合三 MS-TD v3','goal':6144,'format':'NTD1 / Float32 LE','bytes':len(score.read_bytes()),'sha256':sel['scoreSha256'],'candidate':sel['winner'],'stages':[1536,3072],'additionalSelectedTrainingEpisodes':200000+(100000 if sel['winner']!='a' else 0),'protocol':'training/v3/protocol.json','selection':sel,'ruleChanges':False,'searchBudgetChanges':False};meta.update({'version':3,'kind':kind,'baseSha256':json.loads((d/'models/ntuple-v2.json').read_text())['sha256'],'cumulativeTrainingEpisodes':1490000+meta['additionalSelectedTrainingEpisodes'],'search':{'maxDepth':3,'maxNodes':24000,'browserSoftBudgetMs':160}});(d/'models/ntuple-v3.json').write_text(json.dumps(meta,indent=2)+'\n');print(config)
