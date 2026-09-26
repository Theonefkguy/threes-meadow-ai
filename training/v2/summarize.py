"""Summarize completed paired evaluations without optional stopping."""
import argparse,json,math,hashlib,statistics
from pathlib import Path

def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]
def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [mid-half,mid+half]
def summarize(rows):
    n=len(rows);wins=sum(r['success'] for r in rows)
    return {'games':n,'successes':wins,'rate':wins/n,'wilson95':wilson(wins,n),
      'truncated':sum(r['truncated'] for r in rows),'meanMovesToStop':statistics.mean(r['moves'] for r in rows),
      'meanNativeMsPerMove':1000*sum(r['seconds'] for r in rows)/sum(r['moves'] for r in rows),
      'depthCounts':[sum(r['depthCounts'][i] for r in rows) for i in range(3)]}
def paired(a,b):
    da={x['seed']:x for x in a};db={x['seed']:x for x in b};assert da.keys()==db.keys()
    diff=[int(db[s]['success'])-int(da[s]['success']) for s in sorted(da)];n=len(diff)
    only_a=diff.count(-1);only_b=diff.count(1);delta=statistics.mean(diff);se=statistics.stdev(diff)/math.sqrt(n)
    discordant=only_a+only_b;m=min(only_a,only_b)
    p=min(1.,2*sum(math.comb(discordant,i) for i in range(m+1))/2**discordant) if discordant else 1
    return {'difference':delta,'pairedNormal95':[delta-1.959964*se,delta+1.959964*se],
      'v1Only':only_a,'v2Only':only_b,'mcnemarExactTwoSidedP':p}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('v1');parser.add_argument('v2');parser.add_argument('out');args=parser.parse_args()
    a,b=read(args.v1),read(args.v2);assert len(a)==len(b)==1000
    assert not any(r['truncated'] for r in a+b)
    result={'v1':summarize(a),'v2':summarize(b),'paired':paired(a,b)}
    Path(args.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
