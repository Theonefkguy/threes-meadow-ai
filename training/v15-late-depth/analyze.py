from pathlib import Path
from collections import defaultdict
import json,numpy as np
from scipy import stats
p=Path('training/v15-late-depth')
rows=[json.loads(x) for x in (p/'results-raw.jsonl').read_text().splitlines()]
assert len(rows)==1024
by=defaultdict(dict)
for x in rows:
 key=(x['index'],x['replicate']);assert x['depth'] not in by[key];by[key][x['depth']]=x
assert len(by)==512 and all(set(x)=={4,5} for x in by.values())
def interval(values,scale=100):
 x=np.asarray(values,float);mean=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(len(x)));radius=float(stats.t.ppf(.975,len(x)-1))*se if se else 0
 return {'difference':scale*mean,'CI95':[scale*(mean-radius),scale*(mean+radius)],'p':float(stats.ttest_1samp(x,0).pvalue) if se else (1. if mean==0 else 0.),'clusters':len(x)}
per=[]
for i in range(128):
 pairs=[by[(i,r)] for r in range(4)]
 d1536=np.mean([int(x[5]['reached1536'])-int(x[4]['reached1536']) for x in pairs])
 d3072=np.mean([int(x[5]['reached3072'])-int(x[4]['reached3072']) for x in pairs])
 per.append({'index':i,'depth5Minus4Reached1536':float(d1536),'depth5Minus4Reached3072':float(d3072),'interaction':float(d3072-d1536)})
result={'snapshots':128,'replicates':4,'continuations':1024,'arms':{},'effects':{}}
for d in [4,5]:
 arm=[x for x in rows if x['depth']==d];moves=sum(x['moves'] for x in arm);cpu=sum(x['searchCPU'] for x in arm)
 result['arms'][str(d)]={'reached1536':sum(x['reached1536'] for x in arm),'reached3072':sum(x['reached3072'] for x in arm),'runs':len(arm),'reached1536Percent':100*np.mean([x['reached1536'] for x in arm]),'reached3072Percent':100*np.mean([x['reached3072'] for x in arm]),'meanCPUSecondsPerRun':cpu/len(arm),'meanCPUSecondsPerMove':cpu/moves,'maxMoveCPUSeconds':max(x['maxMoveCPU'] for x in arm),'meanMoves':moves/len(arm),'nodesPerMove':sum(x['nodes'] for x in arm)/moves}
result['effects']['reached1536']=interval([x['depth5Minus4Reached1536'] for x in per])
result['effects']['reached3072']=interval([x['depth5Minus4Reached3072'] for x in per])
result['effects']['higherTargetInteraction']=interval([x['interaction'] for x in per])
result['CPUPerMoveRatio5to4']=result['arms']['5']['meanCPUSecondsPerMove']/result['arms']['4']['meanCPUSecondsPerMove']
(p/'per-snapshot.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in per))
(p/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
