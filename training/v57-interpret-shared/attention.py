"""V57 step 2: where each attention head looks, per view (the board turned so that the judged move is left).
The legal views of 4096 validation positions. "Lift" as V51: average attention weight per key of a
category x 20 (1 = uniform).
- summary token -> token kinds, -> cells by content (largest, second largest, other >= 3, 1/2, empty) and
  by whether the card moves when this view pushes left;
- cell -> cell (non-empty queries) by geometry (self, left / right neighbour, up-down neighbour, same row
  at distance 2-3, same column at distance 2-3, elsewhere) and by content (merge partner to the left /
  right / above-below / not adjacent, other cards, empty cells);
- cell -> preview token.  -> attention.json, fig/attention-lift.png"""
import json
import numpy as np
import torch
from common import load_net, sample, to_dev, run, views, rotate, plt_setup, L, dev
import features as F49

net = load_net(); a = sample(); N = 4096
x = to_dev(a, slice(0, N)); _, _, _, _, att, _ = run(net, x, keep_attn=True)
v, legal = views(net, x); b = v['board'][legal]                                                    # [n,16]
moved = torch.zeros_like(b, dtype=torch.bool)
for k, bb in enumerate(b.cpu().numpy().astype(int).tolist()):
    for i in F49.move(bb, 0)[1]: moved[k, i] = True
def mergeable(p, q): return (p > 0) & (q > 0) & (((p + q) == 3) | ((p >= 3) & (p == q)))
R = torch.arange(16, device=dev) // 4; C = torch.arange(16, device=dev) % 4
dr = R[:, None] - R[None]; dc = C[:, None] - C[None]                                                # query - key
dist = dr.abs() + dc.abs()
geo = {'自己': dist == 0, '左邻': (dr == 0) & (dc == 1), '右邻': (dr == 0) & (dc == -1), '上下邻': (dc == 0) & (dr.abs() == 1),
       '同行 距离2–3': (dr == 0) & (dc.abs() >= 2), '同列 距离2–3': (dc == 0) & (dr.abs() >= 2), '其他位置': (dr != 0) & (dc != 0)}
srt = b.sort(1, descending=True).values; second = srt[:, 1:2]
first_max = torch.zeros_like(b, dtype=torch.bool).scatter_(1, b.argmax(1, keepdim=True), True)
sec = (b == second) & ~first_max & (second > 0)
cont_sum = {'最大牌': first_max, '第二大的牌': sec, '其他 ≥3 的牌': (b >= 3) & ~first_max & ~sec, '1 和 2': (b == 1) | (b == 2), '空格': b == 0,
            '往左会动的牌': moved & (b > 0), '不动的牌': ~moved & (b > 0)}
pair = mergeable(b[:, :, None], b[:, None, :])
nonempty_q = (b > 0)[:, :, None].expand(-1, -1, 16)
cont_cell = {'可合并 在左边': pair & geo['左邻'], '可合并 在右边': pair & geo['右邻'], '可合并 上下': pair & geo['上下邻'], '可合并 不相邻': pair & (dist > 1),
             '其他牌（非自己）': ~pair & (b > 0)[:, None, :] & (dist > 0), '空格': (b == 0)[:, None, :].expand(-1, 16, -1)}
def lift(w, mask, qmask=None):
    m = mask.float() if qmask is None else (mask & qmask).float()
    return float((w * m).sum() / m.sum().clamp(min=1) * 20)
rows = []
for li, w in enumerate(att):
    w = w[legal]
    for h in range(w.shape[1]):
        ws = w[:, h]; s = ws[:, 0]
        r = {'layer': li + 1, 'head': h + 1,
             'summary': {'汇总': float(s[:, 0].mean() * 20), '格子': float(s[:, 1:17].mean() * 20), '提示牌': float(s[:, 17].mean() * 20),
                         '剩余张数': float(s[:, 18].mean() * 20), '奖励周期': float(s[:, 19].mean() * 20)},
             'summaryCells': {k: float((s[:, 1:17] * m).sum() / m.sum() * 20) for k, m in cont_sum.items()}}
        cc = ws[:, 1:17, 1:17]
        r['cellGeometry'] = {k: lift(cc, m[None].expand(len(b), -1, -1), nonempty_q) for k, m in geo.items()}
        r['cellContent'] = {k: lift(cc, m, nonempty_q) for k, m in cont_cell.items()}
        r['cellToPreview'] = float(ws[:, 1:17, 17][(b > 0)].mean() * 20)
        rows.append(r)
(L / 'attention.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1) + '\n')
for r in rows:
    print(r['layer'], r['head'], {k: round(v, 1) for k, v in r['summary'].items()}, {k: round(v, 1) for k, v in r['summaryCells'].items()})
    print('     ', {k: round(v, 1) for k, v in r['cellGeometry'].items()}, {k: round(v, 1) for k, v in r['cellContent'].items()}, 'preview', round(r['cellToPreview'], 1))

plt = plt_setup()
groups = [('summary', '汇总 token 看哪类 token'), ('summaryCells', '汇总 token 看哪些格子'), ('cellGeometry', '格子看格子：位置'), ('cellContent', '格子看格子：内容')]
fig, ax = plt.subplots(1, 4, figsize=(20, 6.5), gridspec_kw={'width_ratios': [5, 7, 7, 6]})
labels = [f'层{r["layer"]} 头{r["head"]}' for r in rows]
for k, (g, title) in enumerate(groups):
    keys = list(rows[0][g]); M = np.array([[r[g][q] for q in keys] for r in rows])
    ax[k].imshow(np.log2(np.clip(M, 1 / 16, 16)), cmap='RdBu_r', vmin=-4, vmax=4, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): ax[k].text(j, i, f'{M[i, j]:.1f}', ha='center', va='center', fontsize=8)
    ax[k].set_xticks(range(len(keys)), keys, rotation=40, ha='right'); ax[k].set_yticks(range(len(rows)), labels if k == 0 else [''] * len(rows))
    ax[k].set_title(title, fontsize=11)
fig.suptitle('注意力的倍数（每个视角都是"往左推"；1 = 均匀，红 = 多看，蓝 = 少看）', fontsize=12)
fig.savefig(L / 'fig' / 'attention-lift.png'); plt.close(fig)
