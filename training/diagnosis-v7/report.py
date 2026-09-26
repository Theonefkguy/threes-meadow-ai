from pathlib import Path
import json,math,statistics,hashlib
from scipy.stats import binomtest
p=Path('training/diagnosis-v7');old=Path('training/v7')
def read(f):return [json.loads(x) for x in f.read_text().splitlines()]
rows=json.loads((p/'fit-data.json').read_text()); snaps=read(p/'snapshots.jsonl');assess=read(old/'assessment.jsonl')
result={'modelSHA256':hashlib.sha256((old/'base.ntd').read_bytes()).hexdigest(),'sampleSplit':json.loads((p/'fit-split.json').read_text()),'fit':{},'crossValidation':{},'teacherActions':{}}
for f in sorted(p.glob('fit-*.jsonl')):
 data=read(f);output=[]
 for r in data:
  by={}
  for x,pred in zip(rows,r['predictions']):by.setdefault(x['source'],[]).append((x,pred))
  agreement=[0,0];totals=[0,0]
  for source,items in by.items():
   rewards=snaps[source]['rewards'];target=max(items,key=lambda pair:pair[0]['target']+rewards[pair[0]['direction']])[0]['direction'];chosen=max(items,key=lambda pair:pair[1]+rewards[pair[0]['direction']])[0]['direction'];split=items[0][0]['split'];agreement[split]+=chosen==target;totals[split]+=1
  output.append({k:v for k,v in r.items() if k!='predictions'}|{'onePlyTeacherAgreement':[a/n for a,n in zip(agreement,totals)],'note':'one-ply proxy for fitting only, not deployed3-ply policy accuracy'})
 result['fit'][f.stem]=output
for seed in [202609221,202609222,202609223]:
 combined=[]
 for epoch in [0,1,10,100]:
  trainSq=0;testSq=0;ntrain=0;ntest=0;drift=[];folds=[]
  for fold in range(5):
   data=[x.split() for x in (p/f'cv-data-{fold}.txt').read_text().splitlines()];r=next(x for x in read(p/f'cv-{fold}-{seed}.jsonl') if x['epoch']==epoch);n=sum(int(x[3]) for x in data);testSq+=r['rmse'][1]**2*n;ntest+=n;trainSq+=r['rmse'][0]**2*(len(data)-n);ntrain+=len(data)-n;drift.append(r['normalDriftRMSE']);folds.append(r['rmse'][1])
  combined.append({'epoch':epoch,'trainRMSE':math.sqrt(trainSq/ntrain),'outOfFoldRMSE':math.sqrt(testSq/ntest),'normalDriftRMSEMean':statistics.mean(drift),'heldoutFoldRMSE':folds})
 result['crossValidation'][str(seed)]=combined
for key in ['teacherAction','exact3Action','exact4Action']:
 changes=[];deltas=[]
 for a,s in zip(assess,snaps):
  action=a[key] if key in a else s[key];base=a['baselineAction']
  if action!=base:changes.append(a['index']);deltas.append((a['wins'][action]-a['wins'][base])/a['trialsPerAction'])
 result['teacherActions'][key]={'changed':len(changes),'of':256,'meanRolloutDeltaAllStates':sum(deltas)/256,'meanRolloutDeltaChangedStates':statistics.mean(deltas),'caution':'exploratory adaptive8/24 rollouts with baseline continuation; selected failed-game states; not full teacher game success'}
# Pure fit uses frozen old teacher labels. The original V7 candidates saw ALL labels.
result['limits']=['Original candidate fit audit is in-sample, even on the later diagnostic split. Only new pure fits exclude heldout source games.','Three seeds only vary supervised sample ordering; they are not three independent online RL trainings.','Cross-validation is secondary sensitivity analysis, no inference of game success from RMSE alone.','Current game simulator rules and search assumptions are held fixed.','512-game diagnostic cannot establish equivalence or reliably detect small gains.','No new online replay-dose, depth-matched RL, auxiliary-goal or early-state curriculum experiment in this diagnostic.']
if all((p/f'{m}.jsonl').exists() for m in ['current3','exact3','exact4']):
 games={m:read(p/f'{m}.jsonl') for m in ['current3','exact3','exact4']};result['games']={};result['comparisons']={}
 for m,rs in games.items():
  assert len(rs)==512 and [r['seed'] for r in rs]==list(range(9510001,9510513))
  if m!='current3':assert all(sum(r['depths'])==r['depths'][int(m[-1])-1] for r in rs)
  result['games'][m]={'wins':sum(r['success'] for r in rs),'n':len(rs),'rate':statistics.mean(r['success'] for r in rs),'meanCPUSeconds':statistics.mean(r['cpu'] for r in rs),'depthCounts':[sum(r['depths'][i] for r in rs) for i in range(4)]}
 for a,b in [('exact4','exact3'),('exact3','current3'),('exact4','current3')]:
  x=games[a];y=games[b];d=[int(i['success'])-int(j['success']) for i,j in zip(x,y)];delta=statistics.mean(d);se=statistics.stdev(d)/math.sqrt(len(d));plus=d.count(1);minus=d.count(-1)
  result['comparisons'][a+'-'+b]={'difference':delta,'paired95CI':([delta-1.96*se,delta+1.96*se] if plus+minus else [-(1-.05**(1/len(d))),1-.05**(1/len(d))]),'intervalMethod':('paired normal approximation' if plus+minus else 'exact zero-discordance upper bound on absolute difference'),'onlyA':plus,'onlyB':minus,'mcnemarExactP':binomtest(plus,plus+minus).pvalue if plus+minus else 1,'secondaryExceptExact4MinusExact3':a+'-'+b!='exact4-exact3'}
if (p/'interference.jsonl').exists():
 probe=read(p/'interference.jsonl');result['interference']={}
 for ratio in [0,.0021,.02,.1]:
  final=[r for r in probe if r['ratio']==ratio and r['epoch']==5]
  initial=[r for r in probe if r['ratio']==ratio and r['epoch']==0]
  assert len(final)==len(initial)==3
  result['interference'][str(ratio)]={'initialTrainRMSE':statistics.mean(r['rmse'][0] for r in initial),'finalTrainRMSE':statistics.mean(r['rmse'][0] for r in final),'initialHeldoutRMSE':statistics.mean(r['rmse'][1] for r in initial),'finalHeldoutRMSE':statistics.mean(r['rmse'][1] for r in final),'TDUpdates':final[0]['TDUpdates'],'teacherUpdates':final[0]['teacherUpdates']}
 result['interferenceCaveat']='Fixed-replay interference probe after100-epoch supervised fit:100 fresh normal2-ply baseline games, frozen5-step lambdaTD targets,5passes,TDalpha=.003,teacherAlpha=.001. SameTD order across dose arms within seed. Not online RL, not1536 success rates.'
if (p/'context.jsonl').exists():
 context=read(p/'context.jsonl')
 result['contextAliasing']={'sourceBoards':len({r['source'] for r in context}),'afterstates':len(context),'medianTargetRange':statistics.median(r['range'] for r in context),'meanTargetRange':statistics.mean(r['range'] for r in context),'maximumTargetRange':max(r['range'] for r in context),'contextOnlyIrreducibleRMSE':math.sqrt(statistics.mean(r['variance'] for r in context)),'largestExample':max(context,key=lambda r:r['range']),'interpretation':'Completed3-ply score targets under alternative feasible normal preview draws from the same pre-draw bag counts. Each fixed afterstate has one board-only prediction but multiple conditional targets. RMSE lower bound is for this synthetic weighted context dataset only, not a bound on original teacher-label RMSE or game success.'}
(p/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['fit','limits']},indent=2))
