from pathlib import Path
from collections import defaultdict,Counter
import json,numpy as np
from scipy import stats
p=Path('training/v12-teacher');states=[json.loads(l) for l in (p/'states.jsonl').read_text().splitlines()];rolls=[json.loads(l) for l in (p/'rollouts.jsonl').read_text().splitlines()];by=defaultdict(list)
for r in rolls:by[r['seed']].append(r)
assert [s['seed'] for s in states]==list(range(11710001,11710257))
differences=[];conditional=[];score=[];perstate=[]
for s in states:
 rs=by[s['seed']];different=s['teacherAction']!=s['baseAction'];assert len(rs)==(32 if different else 0)
 if different:
  assert sorted(r['replicate'] for r in rs)==list(range(32))
  diff=float(np.mean([int(r['teacherWin'])-int(r['baseWin']) for r in rs]));conditional.append(diff);score.append(float(np.mean([r['teacherReward']-r['baseReward'] for r in rs])))
 else:diff=0
 differences.append(diff);perstate.append({**s,'teacherWins':sum(r['teacherWin'] for r in rs) if different else None,'baseWins':sum(r['baseWin'] for r in rs) if different else None,'replicates':len(rs),'difference':diff})
def ci(xs,scale=100):
 x=np.array(xs,float);mean=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else 0;radius=float(stats.t.ppf(.975,len(x)-1))*se if len(x)>1 else 0
 return {'mean':scale*mean,'CI95':[scale*(mean-radius),scale*(mean+radius)],'p':float(stats.ttest_1samp(x,0).pvalue) if se else (1. if mean==0 else None),'clusters':len(x)}
r={'sourceStates':len(states),'disagreementStates':len(conditional),'agreementStates':len(states)-len(conditional),'disagreementPercent':100*len(conditional)/len(states),'pairedReplicates':len(rolls),'continuations':2*len(rolls),'primaryAllStatesPP':ci(differences),'disagreementOnlyPP':ci(conditional) if conditional else None,'disagreementTeacherWins':sum(r['teacherWin'] for r in rolls),'disagreementBaseWins':sum(r['baseWin'] for r in rolls),'disagreementTeacherPercent':100*np.mean([r['teacherWin'] for r in rolls]) if rolls else None,'disagreementBasePercent':100*np.mean([r['baseWin'] for r in rolls]) if rolls else None,'statesWithHigherTeacherEstimate':sum(x>0 for x in conditional),'statesWithLowerTeacherEstimate':sum(x<0 for x in conditional),'statesWithTiedEstimate':sum(x==0 for x in conditional),'scoreDifferenceExploratory':ci(score,1) if score else None,'meanTeacherDecisionCPUms':1000*np.mean([s['teacherCPU'] for s in states]),'meanBaseDecisionCPUms':1000*np.mean([s['baseCPU'] for s in states]),'rankCounts':dict(Counter(s['rank'] for s in states))}
(p/'per-state.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in perstate));(p/'results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
