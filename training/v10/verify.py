from pathlib import Path
import hashlib,json,gzip,numpy as np
p=Path('training/v10');m=json.loads((p/'manifest.json').read_text());base=Path('training/v7/base.ntd').read_bytes();boundary=16+8*65536*4
for n,meta in m['models'].items():
 data=gzip.decompress((p/n/'checkpoint.ntd.gz').read_bytes());assert hashlib.sha256(data).hexdigest()==meta['sha256'];assert data[boundary:]==base[boundary:];assert data[:boundary]!=base[:boundary];assert np.isfinite(np.frombuffer(data[16:],dtype='<f4')).all();assert 120<=meta['training']['cpuSeconds']<122;assert meta['training']['truncated']==0
boards=[]
for l in (p/'pool-regular.txt').read_text().splitlines():
 fields=l.split();board=list(map(int,fields[:16]));assert max(board)==11;left=int(fields[16]);deck=list(map(int,fields[17:17+left]));counts=list(map(int,fields[-3:]));assert counts==[deck.count(i) for i in (1,2,3)];boards.append(board)
source_seeds={json.loads(l)['seed'] for l in (p/'pool-games.jsonl').read_text().splitlines()}
assert source_seeds==set(range(11210001,11210129)) and not source_seeds.intersection(range(11310001,11310513))
out={'modelsVerified':len(m['models']),'finiteWeights':True,'lateStagesBitIdentical':True,'earlyStageChanged':True,'noTruncatedTrainingGames':True,'poolStates':len(boards),'poolDeckCountsMatchPublicCounts':True,'teacherPoolDisjointFromEvaluationSeeds':True}
(p/'verification.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
