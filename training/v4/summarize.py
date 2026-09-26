from pathlib import Path
import json,math
from scipy.stats import binomtest
v=Path('training/v4');load=lambda n:[json.loads(l) for l in (v/(n+'.jsonl')).read_text().splitlines()]
def wilson(k,n):
 z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [c-h,c+h]
def stat(rows):
 n=len(rows);k=sum(r['success'] for r in rows);m=sum(r['maxRank']>=13 for r in rows)
 return {'games':n,'wins6144':k,'rate6144':k/n,'wilson95':wilson(k,n),'reached3072':m,'rate3072':m/n,'rate6144Given3072':k/m,'truncated':sum(r['truncated'] for r in rows),'meanMoves':sum(r['moves'] for r in rows)/n}
a,b=load('holdout-v3'),load('holdout-v4');assert len(a)==len(b)==5000;assert [r['seed'] for r in a]==[r['seed'] for r in b]==list(range(6410001,6415001));assert not any(r['truncated'] for r in a+b)
x=sum(r['success'] and not s['success'] for r,s in zip(a,b));y=sum(s['success'] and not r['success'] for r,s in zip(a,b));diff=(y-x)/5000;se=math.sqrt(((x+y)/5000-diff*diff)/5000);ci=[diff-1.96*se,diff+1.96*se]
r={'v3':stat(a),'v4':stat(b),'paired':{'difference':diff,'difference95':ci,'v3Only':x,'v4Only':y,'exactMcNemarP':binomtest(min(x,y),x+y,.5).pvalue if x+y else 1},'selection':json.loads((v/'selection.json').read_text()),'statisticalPromotionGate':ci[0]>0};(v/'evaluation.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
