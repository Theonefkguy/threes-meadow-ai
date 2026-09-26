from pathlib import Path
import json,hashlib,shutil,datetime
from summarize import read,summarize
root=Path(__file__).resolve().parents[2]
rows={name:read(root/f'training/v2/screen-{name}.jsonl') for name in ['v1','a','b','c']}
for name,data in rows.items():
    assert len(data)==200 and {r['seed'] for r in data}==set(range(910001,910201)),name
    assert not any(r['truncated'] for r in data),name
screen={name:summarize(data) for name,data in rows.items()}
winner=max(['a','b','c'],key=lambda name:screen[name]['successes'])
paths={'a':root/'training/v2/run-a/score-400000.ntd','b':root/'training/v2/run-b-guided/goal-100000.goal','c':root/'training/v2/run-c-guided/goal-100000.goal'}
selection={'frozenAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'winner':winner,'screen':screen,
 'hashes':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()},
 'v1Hash':hashlib.sha256((root/'dist/models/ntuple-v1.bin').read_bytes()).hexdigest(),
 'holdoutInspected':False,'modelEpisodes':400000 if winner=='a' else 100000}
if winner!='a' and screen[winner]['successes']>screen['a']['successes'] and screen[winner]['successes']>screen['v1']['successes']:
    raise SystemExit('Goal pilot wins: protocol calls for further training before final selection.')
if screen[winner]['successes']<=screen['v1']['successes']:
    raise SystemExit('No candidate beats V1: retain baseline and investigate before promotion.')
(root/'training/v2/selection.json').write_text(json.dumps(selection,indent=2)+'\n')
if winner=='a':
    shutil.copyfile(paths[winner],root/'dist/models/ntuple-v2.bin');config={'kind':'score','scoreVersion':'v2'}
else:
    shutil.copyfile(paths[winner],root/'dist/models/goal-v2.bin');config={'kind':'goal','scoreVersion':'v1'}
(root/'dist/models/v2-policy.json').write_text(json.dumps(config,indent=2)+'\n')
print(json.dumps(selection,indent=2))
