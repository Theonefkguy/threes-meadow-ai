from pathlib import Path
import json, math
from scipy.stats import binomtest
v=Path('training/v6');s=json.loads((v/'selection.json').read_text())
def wilson(k,n):
 z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [c-h,c+h]
def stat(rows):
 n=len(rows);counts={str(target):sum(r['maxRank']>=rank for r in rows) for target,rank in [(1536,12),(3072,13),(6144,14)]}
 return {'games':n,'counts':counts,'rates':{t:k/n for t,k in counts.items()},'wilson95':{t:wilson(k,n) for t,k in counts.items()},'rate3072Given1536':counts['3072']/counts['1536'],'rate6144Given1536':counts['6144']/counts['1536'],'truncated':sum(r['truncated'] for r in rows),'meanMoves':sum(r['moves'] for r in rows)/n,'failuresByMaximum':{str(3*2**(rank-3)):sum(r['maxRank']==rank for r in rows) for rank in range(3,12)}}
def paired(a,b,rank):
 x=sum(r['maxRank']>=rank and t['maxRank']<rank for r,t in zip(a,b));y=sum(t['maxRank']>=rank and r['maxRank']<rank for r,t in zip(a,b));n=len(a);d=(y-x)/n;se=math.sqrt(((x+y)/n-d*d)/n)
 return {'difference':d,'difference95':[d-1.96*se,d+1.96*se],'aOnly':x,'bOnly':y,'exactMcNemarP':binomtest(min(x,y),x+y,.5).pvalue if x+y else 1}
rows={}
for name in ['v4','control']+(['selected'] if s['selected']!='control' else []):
 r=[json.loads(l) for l in (v/f'holdout-{name}.jsonl').read_text().splitlines()];assert [x['seed'] for x in r]==list(range(8410001,8415001));assert not any(x['truncated'] for x in r);rows[name]=r
selected='control' if s['selected']=='control' else 'selected';comparisons={}
for b in rows:
 if b!='v4':comparisons['v4_vs_'+b]={str(t):paired(rows['v4'],rows[b],rank) for t,rank in [(1536,12),(3072,13),(6144,14)]}
if selected!='control':comparisons['control_vs_selected']={str(t):paired(rows['control'],rows['selected'],rank) for t,rank in [(1536,12),(3072,13),(6144,14)]}
p=comparisons['v4_vs_'+selected];early=p['1536']['difference95'][0]>0;late=p['6144']['difference']>=0 and p['6144']['difference95'][0]>-.01
out={'selection':s,'policies':{n:stat(r) for n,r in rows.items()},'comparisons':comparisons,'selectedHoldout':selected,'primaryComparison':'v4_vs_'+selected,'earlyGate':early,'lateGate':late,'statisticalPromotionGate':early and late,'intervalMethod':'Paired normal approximation on per-seed binary difference; Wilson intervals for rates; exact McNemar p; secondary comparisons descriptive, not multiplicity-adjusted.'}
(v/'evaluation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
