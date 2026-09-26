from pathlib import Path
import json,numpy as np
from scipy.stats import binomtest
p=Path('training/v14-depth');data={d:[json.loads(l) for l in (p/f'depth{d}.jsonl').read_text().splitlines()] for d in [4,5]};r={'pairs':16,'arms':{}}
for d,rows in data.items():
 assert [x['seed'] for x in rows]==list(range(12010001,12010017));k=sum(x['success'] for x in rows);ci=binomtest(k,16).proportion_ci(confidence_level=.95,method='exact');cpu=sum(x['searchCPU'] for x in rows);moves=sum(x['moves'] for x in rows)
 r['arms'][str(d)]={'wins':k,'games':16,'successPercent':100*k/16,'exactBinomialCI95Percent':[100*ci.low,100*ci.high],'totalSearchCPUSeconds':cpu,'meanCPUSecondsPerGame':cpu/16,'meanCPUmsPerMove':1000*cpu/moves,'maxMoveCPUSeconds':max(x['maxMoveCPU'] for x in rows),'meanMoves':moves/16,'nodesPerMove':sum(x['nodes'] for x in rows)/moves}
only5=sum(b['success'] and not a['success'] for a,b in zip(data[4],data[5]));only4=sum(a['success'] and not b['success'] for a,b in zip(data[4],data[5]));r['paired']={'onlyDepth5Wins':only5,'onlyDepth4Wins':only4,'bothWin':sum(a['success'] and b['success'] for a,b in zip(data[4],data[5])),'bothLose':sum(not a['success'] and not b['success'] for a,b in zip(data[4],data[5])),'differencePP':100*(only5-only4)/16,'exactMcNemarP':binomtest(only5,only5+only4,.5).pvalue if only5+only4 else 1.}
r['CPUPerMoveRatio5to4']=r['arms']['5']['meanCPUmsPerMove']/r['arms']['4']['meanCPUmsPerMove'];r['CPUPerGameRatio5to4']=r['arms']['5']['meanCPUSecondsPerGame']/r['arms']['4']['meanCPUSecondsPerGame'];(p/'results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
