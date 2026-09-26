from pathlib import Path
import json,statistics,math
p=Path('training/v8-search');data={m:[json.loads(l) for l in (p/f'late-{m}.jsonl').read_text().splitlines()] for m in ['full','mcts','v4']}
for rows in data.values():assert [r['seed'] for r in rows]==list(range(10010001,10010513))
r={'protocol':json.loads((p/'late-protocol.json').read_text()),'arms':{},'comparisons':{}}
for m,rows in data.items():r['arms'][m]={str(3*2**(rank-3)):sum(x['maxRank']>=rank for x in rows) for rank in [12,13,14]}
for base in ['full','v4']:
 d=[int(a['maxRank']>=14)-int(b['maxRank']>=14) for a,b in zip(data['mcts'],data[base])];delta=statistics.mean(d);se=statistics.stdev(d)/math.sqrt(512);r['comparisons']['mcts-'+base]={'difference6144':delta,'paired95CI':[delta-1.96*se,delta+1.96*se]}
(p/'late-results.json').write_text(json.dumps(r,indent=2)+'\n')
