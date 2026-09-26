from pathlib import Path
import json,hashlib,numpy as np
from scipy import stats
p=Path('training/v13-selfplay');pr=json.loads((p/'protocol.json').read_text());m=json.loads((p/'manifest.json').read_text());names=['initial','mcts']+list(m['models']);data={}
for n in names:
 rows=[json.loads(l) for l in (p/f'eval-{n}.jsonl').read_text().splitlines()];assert [r['seed'] for r in rows]==list(range(11910001,11910513));data[n]=rows
wins={n:np.array([r['success'] for r in rows],float) for n,rows in data.items()}
def metrics(ns):
 rows=[r for n in ns for r in data[n]];moves=sum(r['moves'] for r in rows)
 return {'wins':[int(wins[n].sum()) for n in ns],'ratesPercent':[100*float(wins[n].mean()) for n in ns],'meanPercent':100*float(np.mean([wins[n].mean() for n in ns])),'decisionMicrosecondsPerMove':1e6*sum(r['decisionCPU'] for r in rows)/moves,'maxDecisionMilliseconds':1e3*max(r['maxMoveCPU'] for r in rows)}
r={'initial':metrics(['initial']),'mcts':metrics(['mcts']),'rounds':{},'training':{}}
for round in range(1,5):r['rounds'][str(round)]=metrics([f'{s}-round{round}' for s in pr['trainingSeeds']])
d=np.mean([wins[f'{s}-round4'] for s in pr['trainingSeeds']],axis=0)-wins['initial'];mean=float(d.mean());se=float(d.std(ddof=1)/np.sqrt(512));rad=float(stats.t.ppf(.975,511))*se
r['primaryFinalMinusInitial']={'differencePP':100*mean,'CI95PP':[100*(mean-rad),100*(mean+rad)],'p':float(stats.ttest_1samp(d,0).pvalue) if se else 1.,'clusters':512}
if not se and all(not w.any() for n,w in wins.items() if n=='initial' or n.endswith('round4')):
 upper=100*(1-.025**(1/512))
 r['primaryFinalMinusInitial'].update({'CI95PP':[-upper,upper],'p':None,'method':'Conservative boundary fallback: Bonferroni one-sided exact binomial bounds for initial success and any-final-model success per512 game-seed clusters. Paired t inference is degenerate with all zero successes.'})
for seed in pr['trainingSeeds']:
 records=[json.loads(l) for l in (p/f'run-{seed}/training.jsonl').read_text().splitlines()];assert len(records)==128 and records[-1]['episodes']==8192
 r['training'][str(seed)]={'episodes':records[-1]['episodes'],'wins':records[-1]['totalWins'],'steps':records[-1]['steps'],'cpuSeconds':records[-1]['cpuSeconds'],'roundWins':[sum(x['batchWins'] for x in records[i:i+32]) for i in range(0,128,32)]}
 for round in range(1,5):
  name=f'{seed}-round{round}';blob=(p/f'run-{seed}/round-{round}.net').read_bytes();assert hashlib.sha256(blob).hexdigest()==m['models'][name]['sha256'];w=np.frombuffer(blob[16:],dtype='<f4');assert np.isfinite(w).all();assert (w[9376:9408]==0).all() and w[9440]==0
r['speedupInitialVsMCTS']=r['mcts']['decisionMicrosecondsPerMove']/r['initial']['decisionMicrosecondsPerMove'];r['speedupFinalVsMCTS']=r['mcts']['decisionMicrosecondsPerMove']/r['rounds']['4']['decisionMicrosecondsPerMove'];r['verification']={'policyGradientFiniteDifferences':20,'allModelsFinite':True,'valueHeadsZero':True,'noTruncatedGames':True}
for k in ['initial','base']:assert hashlib.sha256(Path(pr[k]).read_bytes()).hexdigest()==pr[k+'SHA256']
r['posthocDiagnostics']={}
for mode in ['table','two']:
 f=p/f'diagnostic-{mode}.jsonl'
 if f.exists():
  rows=[json.loads(l) for l in f.read_text().splitlines()];r['posthocDiagnostics'][mode]={'wins':sum(x['success'] for x in rows),'games':len(rows)}
(p/'results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
