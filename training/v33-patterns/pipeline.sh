#!/bin/bash
# V33 step 1 (repository root): nohup caffeinate -i bash training/v33-patterns/pipeline.sh &
set -e
L=training/v33-patterns; mkdir -p $L/data
for pass in A B; do seq -f %.0f 0 100 14955 | xargs -P 16 -n 1 bash $L/one.sh $pass; done
python3 - <<'PY'
import json
P=json.load(open('training/v33-patterns/protocol.json'))
for k,v in P['candidates'].items(): print(k, v if v!='none (base only)' else 'none')
PY
for k in NONE P5 C1 C2 C3; do
  spec=$(python3 -c "import json;v=json.load(open('$L/protocol.json'))['candidates']['$k'];print('none' if v.startswith('none') else v)")
  $L/bin/fit dist/models/ntuple-v23.bin "$spec" $L/data/*.bin -- $L/fit-$k.json &
done
wait
python3 $L/analyze.py > /dev/null
echo done > $L/pipeline.done
