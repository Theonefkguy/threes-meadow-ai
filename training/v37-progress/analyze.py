"""V37 screen analysis -> screen.json (paired by validation start index, like V33/V36)."""
import json, math
from pathlib import Path
L = Path('training/v37-progress'); P = json.loads((L / 'protocol.json').read_text())
KEYS = ['none', 'bias3', 's2', 's23', 's234', 'chain', 'mass']; SIZE = {'bias3': 3, 's2': 6, 's23': 36, 's234': 216, 'chain': 16, 'mass': 41}
fits = {k: json.loads((L / f'fit-{k}.json').read_text()) for k in KEYS}
mse = {k: f['bestValidationMSE'] for k, f in fits.items()}
def paired(x, y):  # z > 0: y better than x
    a, b = fits[x]['perEpisode'], fits[y]['perEpisode']; assert a.keys() == b.keys()
    n = sum(v[1] for v in a.values()); d = [a[k][0] - b[k][0] for k in a]
    m = sum(d) / len(d); se = math.sqrt(sum((v - m) ** 2 for v in d) / (len(d) - 1)) / math.sqrt(len(d))
    return {'mseDifference': sum(d) / n, 'z': m / se if se else 0.0}
NONE = P['v33']['NONEvalidationMSE']
gain = lambda k: (mse['none'] - mse[k]) / (NONE - mse['none'])
res = {'validationMSE': mse, 'bestEpoch': {k: f['bestEpoch'] for k, f in fits.items()},
       'reproduces': {'none = V33 P5': mse['none'] == P['v33']['P5validationMSE'], 'bias3 = V36 bias3': mse['bias3'] == P['v36']['bias3validationMSE']},
       'vsNone': {k: {**paired('none', k), 'relativeGain': gain(k), 'values': SIZE[k]} for k in SIZE},
       'vsBias3': {k: paired('bias3', k) for k in SIZE if k != 'bias3'},
       'biasWeights': {k: fits[k]['biasWeights'] for k in SIZE},
       'counts': {k: fits[k]['trainPerGroup'] for k in SIZE}}
eligible = [k for k in SIZE if res['vsNone'][k]['z'] > 2 and gain(k) >= 0.05]
choice = None
if eligible:
    top = max(gain(k) for k in eligible)
    choice = min((k for k in eligible if gain(k) >= top - 0.005), key=lambda k: (SIZE[k], -gain(k)))
res['decision'] = {'eligible': eligible, 'choice': choice, 'go': choice is not None, 'rule': P['decision']}
(L / 'screen.json').write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps({k: v for k, v in res.items() if k not in ('biasWeights', 'counts')}, indent=1))
