import json,sys,itertools
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
print('samples',len(rows),'d3!=d5',sum(x['d3']!=x['d5'] for x in rows),'d4!=d5',sum(x['d4']!=x['d5'] for x in rows))
best=[]
for e5,l5,m5,e4,l4,m4 in itertools.product(range(2,7),range(1,5),[.0005,.001,.002,.005,.01],range(4,9),range(2,5),[.002,.005,.01,.02,.05]):
 if e4<e5 or l4<l5 or m4<m5:continue
 miss=cost5=cost4=0
 for x in rows:
  depth=5 if x['empty']<=e5 or x['legal']<=l5 or x['margin3']<=m5 else 4 if x['empty']<=e4 or x['legal']<=l4 or x['margin3']<=m4 else 3
  pred=x[f'd{depth}'];miss+=pred!=x['d5'];cost5+=depth==5;cost4+=depth==4
 best.append((miss,cost5*4+cost4,cost5,cost4,e5,l5,m5,e4,l4,m4))
for x in sorted(best)[:20]:print(x)
