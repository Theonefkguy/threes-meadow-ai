from pathlib import Path
import json,numpy as np
from scipy import stats
p=Path('training/v9'); m=json.loads((p/'manifest.json').read_text()); protocol=json.loads((p/'protocol.json').read_text())
arms=list(protocol['arms']); seeds=protocol['trainingSeeds']; allrows={}
for name in ['base']+list(m['models']):
 rows=[json.loads(l) for l in (p/f'eval-{name}.jsonl').read_text().splitlines()]
 assert [r['seed'] for r in rows]==list(range(11110001,11110513))
 allrows[name]=rows
wins={k:np.array([r['success'] for r in rows],float) for k,rows in allrows.items()}
def interval(x):
 mean=float(x.mean()); se=float(x.std(ddof=1)/np.sqrt(len(x))); radius=float(stats.t.ppf(.975,len(x)-1))*se
 return {'differencePP':100*mean,'CI95PP':[100*(mean-radius),100*(mean+radius)],'p':float(stats.ttest_1samp(x,0).pvalue) if se else (1. if mean==0 else 0.)}
result={'cohortGames':512,'trainingSeeds':seeds,'baseline':{},'arms':{},'factorial':{}}
for arm in ['base']+arms:
 names=['base'] if arm=='base' else [f'{arm}-{s}' for s in seeds]
 rows=[r for name in names for r in allrows[name]]; moves=sum(r['moves'] for r in rows)
 d={'successCounts':[int(wins[n].sum()) for n in names],'ratesPercent':[float(wins[n].mean()*100) for n in names],'meanRatePercent':float(np.mean([wins[n].mean()*100 for n in names])),'cpuMsPerMove':1000*sum(r['searchCPU'] for r in rows)/moves,'simulationsPerMove':sum(r['work'] for r in rows)/moves,'maxMoveCPUms':1000*max(r['maxMoveCPU'] for r in rows),'overrun20PercentMoves':100*sum(r['overruns20pct'] for r in rows)/moves}
 if arm=='base': result['baseline']=d
 else:
  d['vsBaselineExploratory']=interval(np.mean([wins[n] for n in names],axis=0)-wins['base'])
  d['validationMean']={metric:float(np.mean([m['models'][n]['validation'][metric] for n in names])) for metric in ['validationRMSE','validationKL','validationHuber']}
  result['arms'][arm]=d
avg={a:np.mean([wins[f'{a}-{s}'] for s in seeds],axis=0) for a in arms}
contrasts={'context':(avg['context_value']+avg['context_joint']-avg['board_value']-avg['board_joint'])/2,'policy':(avg['board_joint']+avg['context_joint']-avg['board_value']-avg['context_value'])/2}
for k,x in contrasts.items(): result['factorial'][k]=interval(x)
order=sorted(contrasts,key=lambda k:result['factorial'][k]['p']); prev=0
for i,k in enumerate(order):
 prev=max(prev,min(1,result['factorial'][k]['p']*(2-i))); result['factorial'][k]['holmP']=prev
result['interactionExploratory']=interval(avg['context_joint']-avg['context_value']-avg['board_joint']+avg['board_value'])
result['initialValidation']=json.loads((p/'board_value-202609231/metrics.jsonl').read_text().splitlines()[0])
result['limitations']=['CIs conditional on these three training seeds; clusters are512 game seeds, not1536 independent games.','CPU-timed search changes trajectories with host speed; compare observed compute as well as success.','Same simulation seed separates search from game RNG but is not perfect paired exogenous spawn coupling once trajectories diverge.','Search distillation of score teacher; not a newly completed iterative on-policy RL loop.','No6144 or browser success evaluation in this round.']
(p/'results.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
