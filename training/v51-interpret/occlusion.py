"""V51 step 4b: occlusion over 200 positions (late game, top card >= 1536; model and AI agree; second-best
direction >= 1% worse). For every cell: set it to every other value (empty, 1, 2, 3, 6 .. the largest
card) and average the drop of the chosen direction's probability. Grouped by what the cell holds and by
whether its card moves under the chosen direction.  -> occlusion.json"""
import json
import numpy as np
import torch
from common import load_net, sample, to_dev, run, gaps, L
from engine import move
import sys; sys.path.insert(0, str(L.parent / 'v49-rules')); import features as F49

net = load_net(); a = sample(); v = a['values']; g = gaps(v)
logits = torch.cat([run(net, to_dev(a, slice(i, i + 4096)))[0] for i in range(0, len(v), 4096)]); pick = logits.argmax(1).cpu().numpy()
gs = np.sort(g, 1); cand = np.flatnonzero((g[np.arange(len(pick)), pick] == 0) & (gs[:, 1] >= 0.01) & (a['phase'] >= 12))
ex = np.random.default_rng(7).choice(cand, 200, replace=False); rows = []
for i in ex:
    b0 = a['board'][i].astype(int); top = b0.max(); d0 = int(pick[i]); boards, owner = [b0], [-1]
    for c in range(16):
        for r in range(top + 1):
            if r != b0[c]: bb = b0.copy(); bb[c] = r; boards.append(bb); owner.append(c)
    n = len(boards); y = {k: np.repeat(a[k][i:i + 1], n, 0) for k in ('cards', 'bonus', 'counts', 'cycle')}
    y['board'] = np.array(boards, np.uint8); y['legal'] = np.array([[bool(move(list(bb), d)[1]) for d in range(4)] for bb in boards])
    p = torch.softmax(run(net, to_dev(y))[0], 1)[:, d0].cpu().numpy(); owner = np.array(owner)
    moved = F49.move(b0.tolist(), d0)[1]; srt = sorted(b0, reverse=True); imax = int(np.argmax(b0))
    for c in range(16):
        r = b0[c]; kind = ('最大牌' if c == imax else '第二大的牌' if r == srt[1] and r >= 3 else '其他 ≥6 的牌' if r >= 4 else '3' if r == 3 else '1 或 2' if r in (1, 2) else '空格')
        rows.append({'kind': kind, 'moves': bool(c in moved) if r else None, 'drop': float((p[0] - p[owner == c]).mean())})
out = {}
for k in ('最大牌', '第二大的牌', '其他 ≥6 的牌', '3', '1 或 2', '空格'):
    sel = [r for r in rows if r['kind'] == k]
    out[k] = {'n': len(sel), 'drop': float(np.mean([r['drop'] for r in sel])),
              'moving': float(np.mean([r['drop'] for r in sel if r['moves']])) if any(r['moves'] for r in sel) else None,
              'still': float(np.mean([r['drop'] for r in sel if r['moves'] is False])) if any(r['moves'] is False for r in sel) else None}
out['所有会移动的格子'] = float(np.mean([r['drop'] for r in rows if r['moves']])); out['所有不动的非空格'] = float(np.mean([r['drop'] for r in rows if r['moves'] is False]))
(L / 'occlusion.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n'); print(json.dumps(out, ensure_ascii=False, indent=1))
