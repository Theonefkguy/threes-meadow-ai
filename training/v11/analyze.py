from pathlib import Path
import json,numpy as np
from scipy import stats
p=Path('training/v11');m=json.loads((p/'manifest.json').read_text());pr=json.loads((p/'protocol.json').read_text());data={}
for n in ['base']+list(m['models']):
 rows=[json.loads(l) for l in (p/f'eval-{n}.jsonl').read_text().splitlines()];assert [r['seed'] for r in rows]==list(range(11510001,11510513));data[n]=rows
win={n:np.array([r['success'] for r in rs],float) for n,rs in data.items()};d=np.mean([win[n] for n in m['models']],axis=0)-win['base'];mean=float(d.mean());se=float(d.std(ddof=1)/np.sqrt(512));rad=float(stats.t.ppf(.975,511))*se
r={'gamesPerModel':512,'primary':{'differencePP':100*mean,'CI95PP':[100*(mean-rad),100*(mean+rad)],'p':float(stats.ttest_1samp(d,0).pvalue) if se else 1},'models':{}}
for n,rs in data.items():
 moves=sum(x['moves'] for x in rs);out={'wins':int(win[n].sum()),'successPercent':100*float(win[n].mean()),'cpuMsPerMove':1000*sum(x['searchCPU'] for x in rs)/moves,'simulationsPerMove':sum(x['work'] for x in rs)/moves,'maxMoveCPUms':1000*max(x['maxMoveCPU'] for x in rs),'overrun20pctRate':sum(x['overruns20pct'] for x in rs)/moves}
 if n!='base':
  initial=json.loads((p/n/'metrics.jsonl').read_text().splitlines()[0]);v=m['models'][n];out.update({'epoch':v['epoch'],'initialValidationKL':initial['validationKL'],'validationKL':v['metrics']['validationKL'],'trainKL':v['metrics']['trainKL']})
 r['models'][n]=out
r['policyMeanPercent']=float(np.mean([r['models'][n]['successPercent'] for n in m['models']]))
games=[json.loads(l) for l in (p/'collection-games.jsonl').read_text().splitlines()];assert [g['seed'] for g in games]==list(range(11410001,11410513));teacher=[l.split() for l in (p/'teacher.txt').read_text().splitlines()]
def canon(board):
 forms=[]
 for s in range(8):
  out=[0]*16
  for i,v in enumerate(board):
   y,x=divmod(i,4)
   if s>=4:x=3-x
   for _ in range(s%4):x,y=3-y,x
   out[4*y+x]=v
  forms.append(tuple(out))
 return min(forms)
train=set();val=set();counts=[0,0]
for f in teacher:
 assert len(f)==32;valid=(int(f[0])-11410001)%5==0;counts[valid]+=1;(val if valid else train).add(canon(list(map(int,f[2:18]))));assert abs(sum(float(f[20+2*i]) for i in range(int(f[18])))-1)<1e-10
r['data']={'games':len(games),'success1536':sum(g['success1536'] for g in games),'examples':len(teacher),'trainExamples':counts[0],'validationExamples':counts[1],'D4BoardOverlap':len(train&val)}
(p/'results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
