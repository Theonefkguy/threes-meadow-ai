"""V57 step 4: what each part does, by removing it (as V51 ablate.py / occlusion.py).
Mean ablation of each head and each MLP (mean over the views of 4096 validation positions), and zeroing
each layer's attention-bias table (layers 1-3; layer 4's table is all zero, it never got a gradient).
Measured on the 30000 validation positions (near / bad5) and in 300 whole games with no search.
Occlusion over 200 late positions (top >= 1536; model and AI agree; second best >= 1% worse): each cell
set to every other value up to the largest card; mean drop of the chosen direction's probability, grouped
by what the cell holds and whether it moves under the chosen direction.  -> ablate.json, fig/ablate.png"""
import json, time
import numpy as np
import torch
from common import load_net, sample, to_dev, run, mean_outputs, logits_all, play, score, gaps, plt_setup, L
from engine import move
import features as F49

net = load_net(); a = sample(); M = mean_outputs(net, a)
parts = [('none',)] + [('head', l, h) for l in range(4) for h in range(4)] + [('mlp', l) for l in range(4)] + [('bias', l) for l in range(3)]
res = []
for p in parts:
    t0 = time.time(); abl = [] if p == ('none',) else [p]
    r = {'part': p, **score(logits_all(net, a, ablate=abl, means=M), a['values']), **play(net, ablate=abl, means=M)}; res.append(r)
    print(p, {k: round(v, 4) for k, v in r.items() if k != 'part'}, f'{time.time() - t0:.0f}s', flush=True)

g = gaps(a['values']); pick = logits_all(net, a).argmax(1).cpu().numpy(); gs = np.sort(g, 1)
cand = np.flatnonzero((g[np.arange(len(pick)), pick] == 0) & (gs[:, 1] >= 0.01) & (a['phase'] >= 12))
rows = []
for i in np.random.default_rng(7).choice(cand, 200, replace=False):
    b0 = a['board'][i].astype(int); top = b0.max(); d0 = int(pick[i]); boards, owner = [b0], [-1]
    for c in range(16):
        for r in range(top + 1):
            if r != b0[c]: bb = b0.copy(); bb[c] = r; boards.append(bb); owner.append(c)
    n = len(boards); y = {k: np.repeat(a[k][i:i + 1], n, 0) for k in ('cards', 'bonus', 'counts', 'cycle')}
    y['board'] = np.array(boards, np.uint8); y['legal'] = np.array([[bool(move(list(bb), d)[1]) for d in range(4)] for bb in boards])
    pr = torch.softmax(run(net, to_dev(y))[0], 1)[:, d0].cpu().numpy(); owner = np.array(owner)
    moved = F49.move(b0.tolist(), d0)[1]; srt = sorted(b0, reverse=True); imax = int(np.argmax(b0))
    for c in range(16):
        r = b0[c]; kind = ('最大牌' if c == imax else '第二大的牌' if r == srt[1] and r >= 3 else '其他 ≥6 的牌' if r >= 4 else '3' if r == 3 else '1 或 2' if r in (1, 2) else '空格')
        rows.append({'kind': kind, 'moves': bool(c in moved) if r else None, 'drop': float((pr[0] - pr[owner == c]).mean())})
occ = {}
for k in ('最大牌', '第二大的牌', '其他 ≥6 的牌', '3', '1 或 2', '空格'):
    sel = [r for r in rows if r['kind'] == k]
    occ[k] = {'n': len(sel), 'drop': float(np.mean([r['drop'] for r in sel])),
              'moving': float(np.mean([r['drop'] for r in sel if r['moves']])) if any(r['moves'] for r in sel) else None,
              'still': float(np.mean([r['drop'] for r in sel if r['moves'] is False])) if any(r['moves'] is False for r in sel) else None}
occ['所有会移动的格子'] = float(np.mean([r['drop'] for r in rows if r['moves']])); occ['所有不动的非空格'] = float(np.mean([r['drop'] for r in rows if r['moves'] is False]))
print(json.dumps(occ, ensure_ascii=False))
(L / 'ablate.json').write_text(json.dumps({'ablation': res, 'occlusion': occ}, ensure_ascii=False, indent=1) + '\n')

plt = plt_setup()
lab = ['无'] + [f'层{l + 1}头{h + 1}' for l in range(4) for h in range(4)] + [f'层{l + 1} MLP' for l in range(4)] + [f'层{l + 1} 偏置表' for l in range(3)]
col = ['#555'] + ['#4c78a8'] * 16 + ['#e45756'] * 4 + ['#54a24b'] * 3
fig, ax = plt.subplots(1, 2, figsize=(13, 7.5), sharey=True); y = np.arange(len(res))[::-1]
for k, (key, xl, lim) in enumerate((('near', '与 AI 接近的比例 (%)', (30, 88)), ('reach3072', '不带搜索 300 局合出 3072 的比例 (%)', (0, 90)))):
    ax[k].barh(y, [r[key] * 100 for r in res], color=col); ax[k].axvline(res[0][key] * 100, color='k', lw=0.8, ls='--'); ax[k].set_xlabel(xl); ax[k].set_xlim(*lim)
    for j, r in enumerate(res): ax[k].text(r[key] * 100 + 0.3, y[j], f"{r[key] * 100:.1f}" if k == 0 else f"{r[key] * 100:.0f}", va='center', fontsize=8)
ax[0].set_yticks(y, lab)
fig.suptitle('均值消融：一个头（蓝）、一层 MLP（红）换成平均输出，或一层注意力偏置表清零（绿）'); fig.savefig(L / 'fig' / 'ablate.png'); plt.close(fig)
