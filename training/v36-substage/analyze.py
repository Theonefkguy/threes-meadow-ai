"""V36 screen analysis -> screen.json (paired by validation start index, like V33)."""
import json, math
from pathlib import Path
L = Path('training/v36-substage'); P = json.loads((L / 'protocol.json').read_text())
fits = {k: json.loads((L / f'fit-{k}.json').read_text()) for k in ('none', 'two', 'three', 'four', 'bias3')}
mse = {k: f['bestValidationMSE'] for k, f in fits.items()}
def paired(x, y):  # z > 0: y better than x
    a, b = fits[x]['perEpisode'], fits[y]['perEpisode']; assert a.keys() == b.keys()
    n = sum(v[1] for v in a.values()); d = [a[k][0] - b[k][0] for k in a]
    m = sum(d) / len(d); se = math.sqrt(sum((v - m) ** 2 for v in d) / (len(d) - 1)) / math.sqrt(len(d))
    return {'mseDifference': sum(d) / n, 'z': m / se if se else 0.0, 'starts': len(d)}
NONE = P['v33']['NONEvalidationMSE']
res = {'validationMSE': mse, 'bestEpoch': {k: f['bestEpoch'] for k, f in fits.items()},
       'reproducesV33P5': mse['none'] == P['v33']['P5validationMSE'],
       'groups': {k: {'train': f['trainPerGroup'], 'validation': f['validationPerGroup']} for k, f in fits.items()}}
res['vsNone'] = {k: {**paired('none', k), 'relativeGain': (mse['none'] - mse[k]) / (NONE - mse['none'])} for k in ('two', 'three', 'four', 'bias3')}
best = min(('two', 'three', 'four'), key=lambda k: mse[k]); v = res['vsNone'][best]
res['decision'] = {'candidate': best, 'go': v['z'] > 2 and v['relativeGain'] >= 0.05, 'rule': P['decision']}
(L / 'screen.json').write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))
