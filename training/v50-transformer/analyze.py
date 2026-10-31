"""V50 summary: per-model validation / human numbers (runs/*/log.jsonl), policy-only reach rates
(results/play-*.json), and how close each student is on the positions it reaches itself:
model dag(R-1) evaluated on dagger round R (its own games, labelled by the teacher, not yet trained on),
split by the highest card on the board.  ->  results/summary.json
  uv run python analyze.py"""
import json
from pathlib import Path
import numpy as np
import torch
from model import Net
from train import tensors, usable, evaluate, DATA, dev
L = Path(__file__).resolve().parent
NAMES = ['base'] + [f'dag{r}' for r in range(1, 7)]
CARD = {9: '192', 10: '384', 11: '768', 12: '1536', 13: '3072'}

def phase_eval(model, d):
    net = Net().to(dev); net.load_state_dict(torch.load(model, map_location=dev))
    ok = usable(d); top = d['board'].max(1); out = {'all': evaluate(net, tensors(d, ok))}
    for r, c in CARD.items():
        m = ok & ((top == r) if r > 9 else (top <= r)) if r < 13 else ok & (top >= r)
        if m.sum() >= 200: out[('≤' if r == 9 else '≥' if r == 13 else '') + c] = evaluate(net, tensors(d, m))
    return out

if __name__ == '__main__':
    res = {'models': {}, 'play': {}, 'onPolicy': {}}
    for n in NAMES:
        res['models'][n] = json.loads((L / 'runs' / n / 'log.jsonl').read_text().splitlines()[-1])
        res['play'][n] = {k: v for k, v in json.loads((L / 'results' / f'play-{n}.json').read_text()).items() if k != 'top'}
    for p in ('random', 'rules'): res['play'][p] = {k: v for k, v in json.loads((L / 'results' / f'play-{p}.json').read_text()).items() if k != 'top'}
    for r in range(1, 7):
        d = dict(np.load(DATA / 'dagger' / f'round-{r}.npz')); prev = NAMES[r - 1]
        res['onPolicy'][f'{prev}@round{r}'] = phase_eval(L / 'runs' / prev / 'model.pt', d)
        print(prev, {k: (v['n'], round(v['near'], 3), round(v['bad5'], 3)) for k, v in res['onPolicy'][f'{prev}@round{r}'].items()}, flush=True)
    (L / 'results' / 'summary.json').write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n')
