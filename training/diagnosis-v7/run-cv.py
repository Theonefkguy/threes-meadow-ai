from pathlib import Path
import json,random,subprocess
p=Path('training/diagnosis-v7');rows=json.loads((p/'fit-data.json').read_text());groups=sorted({r['game'] for r in rows})
assert len(groups)==json.loads((p/'fit-split.json').read_text())['groups'] # no inter-game D4 unions in this dataset
random.Random(202609220).shuffle(groups)
for fold in range(5):
 held=set(groups[fold*20:(fold+1)*20]); data=p/f'cv-data-{fold}.txt'
 data.write_text(''.join(f"{r['source']} {r['direction']} {r['weight']} {int(r['game'] in held)} {r['target']} "+' '.join(map(str,r['board']))+'\n' for r in rows))
 for seed in [202609221,202609222,202609223]:
  subprocess.run(['/tmp/threes-diagnosis-fit','training/v7/base.ntd',str(data),str(p/'normal-anchors.txt'),str(seed),'100',str(p/f'cv-{fold}-{seed}.jsonl')],check=True)
(p/'cv-protocol.json').write_text(json.dumps({'purpose':'secondary sensitivity check after initial fit diagnostic; not a new game-performance test','folds':5,'groupBy':'source game; no cross-game D4-identical examples','shuffleSeed':202609220,'trainingShuffleSeeds':[202609221,202609222,202609223],'epochs':[0,1,10,100],'alpha':.01},indent=2))
