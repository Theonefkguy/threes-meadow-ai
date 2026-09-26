from pathlib import Path
import json,math
from scipy.stats import binomtest
v=Path('training/v7');selection=json.loads((v/'selection.json').read_text())
def wilson(k,n):
 z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return[c-h,c+h]
def stats(rows):
 n=len(rows);counts={str(t):sum(r['maxRank']>=rank for r in rows) for t,rank in [(1536,12),(3072,13),(6144,14)]}
 return {'games':n,'counts':counts,'rates':{t:k/n for t,k in counts.items()},'wilson95':{t:wilson(k,n) for t,k in counts.items()},'rate6144Given1536':counts['6144']/counts['1536'],'rate3072Given1536':counts['3072']/counts['1536'],'truncated':sum(r['truncated'] for r in rows)}
def paired(a,b,rank):
 x=sum(r['maxRank']>=rank and s['maxRank']<rank for r,s in zip(a,b));y=sum(s['maxRank']>=rank and r['maxRank']<rank for r,s in zip(a,b));n=len(a);d=(y-x)/n;se=math.sqrt(((x+y)/n-d*d)/n);return {'difference':d,'difference95':[d-1.96*se,d+1.96*se],'aOnly':x,'bOnly':y,'exactMcNemarP':binomtest(min(x,y),x+y,.5).pvalue if x+y else 1}
rows={}
for name in ['baseline','selected','v4']:
 r=[json.loads(l) for l in (v/f'holdout-{name}.jsonl').read_text().splitlines()];assert [x['seed'] for x in r]==list(range(9410001,9415001));assert not any(x['truncated'] for x in r);rows[name]=r
comparisons={a+'_vs_selected':{str(t):paired(rows[a],rows['selected'],rank) for t,rank in [(1536,12),(3072,13),(6144,14)]} for a in ['baseline','v4']}
gates={}
for name,c in comparisons.items():gates[name]={'early':c['1536']['difference95'][0]>0,'late':c['6144']['difference']>=0 and c['6144']['difference95'][0]>-.01}
out={'selection':selection,'policies':{name:stats(r) for name,r in rows.items()},'comparisons':comparisons,'gates':gates,'statisticalPromotionGate':all(all(g.values()) for g in gates.values()),'intervalMethod':'Paired normal approximation on binary per-seed differences; Wilson intervals for rates; exact McNemar p. 3072 and non-primary comparisons are descriptive, not multiplicity adjusted.'}
(v/'evaluation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
