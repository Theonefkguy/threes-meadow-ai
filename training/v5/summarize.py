from pathlib import Path
import json,math
from scipy.stats import binomtest
v=Path('training/v5');s=json.loads((v/'selection.json').read_text())
def wilson(k,n):
 z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [c-h,c+h]
def stat(rows):
 n=len(rows);k=sum(r['success'] for r in rows);m=sum(r['maxRank']>=13 for r in rows);late=sum(r['lateMoves'] for r in rows);corner=sum(r['cornerMoves'] for r in rows)
 return {'games':n,'wins6144':k,'rate6144':k/n,'wilson95':wilson(k,n),'reached3072':m,'rate3072':m/n,'rate6144Given3072':k/m,'truncated':sum(r['truncated'] for r in rows),'meanMoves':sum(r['moves'] for r in rows)/n,'lateMoves':late,'cornerMoves':corner,'cornerRateLate':corner/late}
def paired(a,b):
 x=sum(r['success'] and not t['success'] for r,t in zip(a,b));y=sum(t['success'] and not r['success'] for r,t in zip(a,b));n=len(a);d=(y-x)/n;se=math.sqrt(((x+y)/n-d*d)/n)
 return {'difference':d,'difference95':[d-1.96*se,d+1.96*se],'aOnly':x,'bOnly':y,'exactMcNemarP':binomtest(min(x,y),x+y,.5).pvalue if x+y else 1}
rows={}
for name in ['v4','control','shaping']:
 r=[json.loads(l) for l in (v/f'holdout-{name}.jsonl').read_text().splitlines()];assert [x['seed'] for x in r]==list(range(7410001,7415001));assert not any(x['truncated'] for x in r);rows[name]=r
selected='control' if s['winner']=='control' else 'shaping';comparisons={a+'_vs_'+b:paired(rows[a],rows[b]) for a,b in [('v4','control'),('v4','shaping'),('control','shaping')]};primary=comparisons['v4_vs_'+selected]
out={'selection':s,'policies':{n:stat(r) for n,r in rows.items()},'comparisons':comparisons,'selectedHoldout':selected,'primaryComparison':'v4_vs_'+selected,'statisticalPromotionGate':primary['difference95'][0]>0,'inferenceNote':'Primary promotion comparison was fixed by screening. Other pairwise intervals are descriptive, not multiplicity-adjusted.'};(v/'evaluation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
