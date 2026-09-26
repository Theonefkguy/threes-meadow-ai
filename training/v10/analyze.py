from pathlib import Path
import json,numpy as np
from scipy import stats
p=Path('training/v10');pr=json.loads((p/'protocol.json').read_text());manifest=json.loads((p/'manifest.json').read_text());names=['base']+list(manifest['models']);data={}
for n in names:
 rows=[json.loads(l) for l in (p/f'eval-{n}.jsonl').read_text().splitlines()];assert [x['seed'] for x in rows]==list(range(11310001,11310513));data[n]=rows
win={n:np.array([r['success'] for r in rs],float) for n,rs in data.items()}
def contrast(x):
 mean=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(len(x)));w=stats.t.ppf(.975,len(x)-1)*se
 return {'differencePP':100*mean,'CI95PP':[100*(mean-w),100*(mean+w)],'p':float(stats.ttest_1samp(x,0).pvalue) if se else 1.}
result={'gamesPerModel':512,'arms':{},'contrasts':{},'conditionalOnTrainingSeeds':True};av={}
for a in ['base','ordinary','mixed']:
 ns=['base'] if a=='base' else [f'{a}-{s}' for s in pr['trainingSeeds']];rs=[r for n in ns for r in data[n]];moves=sum(r['moves'] for r in rs);av[a]=np.mean([win[n] for n in ns],axis=0)
 result['arms'][a]={'wins':[int(win[n].sum()) for n in ns],'ratesPercent':[100*float(win[n].mean()) for n in ns],'meanPercent':100*float(av[a].mean()),'cpuMsPerMove':1000*sum(r['searchCPU'] for r in rs)/moves,'simulationsPerMove':sum(r['work'] for r in rs)/moves,'maxMoveCPUms':1000*max(r['maxMoveCPU'] for r in rs),'overruns20pctRate':sum(r['overruns20pct'] for r in rs)/moves}
for name,x in [('ordinaryMinusBase',av['ordinary']-av['base']),('mixedMinusOrdinary',av['mixed']-av['ordinary'])]:result['contrasts'][name]=contrast(x)
prev=0
for i,k in enumerate(sorted(result['contrasts'],key=lambda k:result['contrasts'][k]['p'])):
 prev=max(prev,min(1,(2-i)*result['contrasts'][k]['p']));result['contrasts'][k]['holmP']=prev
result['mixedMinusBaseExploratory']=contrast(av['mixed']-av['base']);result['training']={k:v['training'] for k,v in manifest['models'].items()}
(p/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
