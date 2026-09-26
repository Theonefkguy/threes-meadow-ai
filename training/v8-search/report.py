from pathlib import Path
import json,math,statistics
from scipy.stats import binomtest
p=Path('training/v8-search')
def read(f):return [json.loads(l) for l in f.read_text().splitlines()]
def summary(rs):
 n=sum(r['moves'] for r in rs)
 return {'games':len(rs),'wins':sum(r['success'] for r in rs),'rate':statistics.mean(r['success'] for r in rs),'moves':n,'meanMoveCPUms':1000*sum(r['searchCPU'] for r in rs)/n,'maxMoveCPUms':1000*max(r['maxMoveCPU'] for r in rs),'overrun20pctFraction':sum(r['overruns20pct'] for r in rs)/n,'depthCounts':[sum(r['depths'][k] for r in rs) for k in range(10)]}
result={'protocol':json.loads((p/'protocol.json').read_text()),'screen':{m:summary(read(p/f'screen-{m}.jsonl')) for m in ['full','beam2','adaptive','mcts']},'selection':json.loads((p/'selection.json').read_text())}
m=result['selection']['selected'];f=p/f'holdout-{m}.jsonl'
if f.exists():
 a,b=read(f),read(p/'holdout-full.jsonl');assert len(a)==len(b)==1024 and [r['seed'] for r in a]==[r['seed'] for r in b]==list(range(9910001,9911025))
 result['holdout']={m:summary(a),'full':summary(b)};d=[int(x['success'])-int(y['success']) for x,y in zip(a,b)];delta=statistics.mean(d);se=statistics.stdev(d)/math.sqrt(len(d));plus=d.count(1);minus=d.count(-1);ci=[delta-1.96*se,delta+1.96*se] if plus+minus else [-(1-.05**(1/len(d))),1-.05**(1/len(d))];ratio=result['holdout'][m]['meanMoveCPUms']/result['holdout']['full']['meanMoveCPUms'];result['comparison']={'difference':delta,'paired95CI':ci,'onlySelected':plus,'onlyFull':minus,'mcnemarExactP':binomtest(plus,plus+minus).pvalue if plus+minus else 1,'measuredMeanCPURatio':ratio,'fairnessGate':ratio<=1.05,'earlyImprovementGate':ci[0]>0,'furtherLateBrowserValidationNeededBeforePromotion':True}
if (p/'late-results.json').exists():
 result['late']=json.loads((p/'late-results.json').read_text())
 la,lb=read(p/'late-mcts.jsonl'),read(p/'late-full.jsonl')
 a,b=read(p/'holdout-mcts.jsonl'),read(p/'holdout-full.jsonl')
 d=[int(x['success'])-int(y['success']) for x,y in zip(a,b)]+[int(x['maxRank']>=12)-int(y['maxRank']>=12) for x,y in zip(la,lb)]
 delta=statistics.mean(d);se=statistics.stdev(d)/math.sqrt(len(d));result['pooledEarlyExploratory']={'games':len(d),'mctsWins':sum(x['success'] for x in a)+sum(x['maxRank']>=12 for x in la),'fullWins':sum(x['success'] for x in b)+sum(x['maxRank']>=12 for x in lb),'difference':delta,'paired95CI':[delta-1.96*se,delta+1.96*se],'mcnemarExactP':binomtest(d.count(1),d.count(1)+d.count(-1)).pvalue,'interpretation':'Secondary cohort did not replicate the primary early gain; pooled estimate is descriptive and includeszero. No stable upgrade claim.'}
for name in ['js-validation','worker-results']:
 if (p/(name+'.json')).exists():result[name]=json.loads((p/(name+'.json')).read_text())
result['websiteDecision']={'experimentalOption':'rl-v8','default':'rl-v4','newWeightsTrained':False,'reason':'Primary1024-game early test positive, but supplemental512-game early/late results do not establish a robust advantage; opt-in research option only.'}
(p/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
