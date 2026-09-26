import json,math
from pathlib import Path
v=Path('training/v3')
def load(n):return [json.loads(l) for l in (v/f'{n}.jsonl').read_text().splitlines()]
def wilson(k,n):
 z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [c-h,c+h]
def summary(rows):
 n=len(rows);k=sum(r['success'] for r in rows);m=sum(r['maxRank']>=13 for r in rows)
 return {'games':n,'reached6144':k,'rate6144':k/n,'wilson95':wilson(k,n),'reached3072':m,'rate3072':m/n,'rate6144Given3072':k/m,'meanMoves':sum(r['moves'] for r in rows)/n,'truncated':sum(r['truncated'] for r in rows)}
a,b=load('holdout-v2'),load('holdout-v3');assert len(a)==len(b)==5000;assert [r['seed'] for r in a]==[r['seed'] for r in b]==list(range(3910001,3915001));assert not any(r['truncated'] for r in a+b)
onlyA=sum(x['success'] and not y['success'] for x,y in zip(a,b));onlyB=sum(y['success'] and not x['success'] for x,y in zip(a,b));n=len(a);d=(onlyB-onlyA)/n;se=math.sqrt(((onlyA+onlyB)/n-d*d)/n)
try:
 from scipy.stats import binomtest
 p=binomtest(min(onlyA,onlyB),onlyA+onlyB,.5).pvalue
except ImportError:p=None
result={'v2':summary(a),'v3':summary(b),'paired':{'v2Only':onlyA,'v3Only':onlyB,'difference':d,'difference95':[d-1.96*se,d+1.96*se],'mcnemarExactP':p},'selection':json.loads((v/'selection.json').read_text())}
result['promote']=d>0 and d-1.96*se>0
(v/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
