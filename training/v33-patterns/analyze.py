"""V33 step 1 analysis: python3 training/v33-patterns/analyze.py -> screen.json (paired, clustered by start index)."""
from pathlib import Path
import json, math
HERE = Path(__file__).resolve().parent; P = json.loads((HERE / 'protocol.json').read_text())
fits = {k: json.loads((HERE / f'fit-{k}.json').read_text()) for k in P['candidates']}
mse = {k: f['bestValidationMSE'] for k, f in fits.items()}
def paired(x, y):
    a, b = fits[x]['perEpisode'], fits[y]['perEpisode']; assert a.keys() == b.keys()
    n = sum(v[1] for v in a.values()); d = [(a[k][0] - b[k][0]) for k in a]  # per start: SSE_x - SSE_y
    m = sum(d) / len(d); se = math.sqrt(sum((v - m) ** 2 for v in d) / (len(d) - 1)) / math.sqrt(len(d))
    return {'mseDifference': sum(d) / n, 'z': m / se if se else 0.0, 'starts': len(d)}
res = {'validationMSE': mse, 'bestEpoch': {k: f['bestEpoch'] for k, f in fits.items()}, 'records': {'train': fits['P5']['train'], 'validation': fits['P5']['validation']}}
res['vsP5'] = {k: {**paired('P5', k), 'relativeGain': (mse['P5'] - mse[k]) / (mse['NONE'] - mse['P5'])} for k in ('C1', 'C2', 'C3')}  # z > 0: k better
res['P5vsNONE'] = paired('NONE', 'P5')
best = min(('C1', 'C2'), key=lambda k: mse[k]); v = res['vsP5'][best]
res['decision'] = {'candidate': best, 'go': v['z'] > 2 and v['relativeGain'] >= 0.05, 'rule': P['decision']}
(HERE / 'screen.json').write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))
