"""V57 step 1: the input embeddings and the attention-bias tables of the shared model.
Cards: 16 rank vectors; principal components, cosine similarities (as V51 embed.py), next to dag14's.
Positions: 8 vectors (column 0-3 x edge / inner row); cosine similarities.
Bias tables: per layer and head a 4 x 7 table over (|row difference| 0-3, column difference -3..3), where
column difference = query column - key column, so +1 = the key is one column to the left of the query
(the side cards move to when this view pushes left).  -> embed.json, fig/embed.png, fig/bias.png"""
import json
import numpy as np
from common import load_net, CARD, c51, plt_setup, L

net = load_net(); E = net.cell.weight.detach().cpu().numpy(); P = net.pos.detach().cpu().numpy()
E50 = c51.load_net().cell.weight.detach().cpu().numpy()
cos = lambda X: (X / np.linalg.norm(X, axis=1, keepdims=True)) @ (X / np.linalg.norm(X, axis=1, keepdims=True)).T
def cards(E):
    Ec = E - E.mean(0); _, S, Vt = np.linalg.svd(Ec, full_matrices=False); pc = Ec @ Vt[:3].T; var = S ** 2 / (S ** 2).sum(); big = list(range(3, 15)); C = cos(E)
    return pc, var, C, {'varianceTop3': var[:3].round(3).tolist(), 'pc1VsRank(3..6144)': round(float(np.corrcoef(pc[big, 0], big)[0, 1]), 3),
                        'pc2VsRank': round(float(np.corrcoef(pc[big, 1], big)[0, 1]), 3),
                        'cosNeighbourRanks(3..6144)': [round(float(C[r, r + 1]), 3) for r in range(3, 14)],
                        'meanCosNeighbour': round(float(np.mean([C[r, r + 1] for r in range(3, 14)])), 3),
                        'meanCosBig(768..6144)': round(float(np.mean([C[i, j] for i in range(8, 15) for j in range(8, 15) if i < j])), 3),
                        'cos1_2': round(float(C[1, 2]), 3), 'cos1_3': round(float(C[1, 3]), 3), 'cos2_3': round(float(C[2, 3]), 3)}
pc, var, C, out_s = cards(E); _, _, _, out_50 = cards(E50)
POSN = [f'列{c + 1}{"边" if e == 0 else "中"}' for c in range(4) for e in range(2)]             # index c * 2 + edge/inner
bias = net.bias.detach().cpu().numpy()                                                          # [layer, head, |dr|, dc + 3]
out = {'cards': out_s, 'cardsDag14': out_50, 'positions': {'names': POSN, 'cos': cos(P).round(3).tolist(), 'norm': np.linalg.norm(P, axis=1).round(2).tolist()},
       'bias': {f'L{l + 1}H{h + 1}': {'table': bias[l, h].round(2).tolist(), 'range': round(float(bias[l, h].max() - bias[l, h].min()), 2),
                                      'max': [int(i) for i in np.unravel_index(bias[l, h].argmax(), (4, 7))], 'min': [int(i) for i in np.unravel_index(bias[l, h].argmin(), (4, 7))]}
                for l in range(4) for h in range(4)}}
(L / 'embed.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
print(json.dumps({'cards': out_s, 'cardsDag14': out_50}, ensure_ascii=False))
for k, v in out['bias'].items(): print(k, 'range', v['range'], 'max at', v['max'], 'min at', v['min'])

plt = plt_setup()
fig, ax = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={'width_ratios': [5, 5, 4]})
cm = plt.get_cmap('viridis')
for r in range(15):
    ax[0].scatter(pc[r, 0], pc[r, 1], s=60, color=cm(r / 14) if r >= 3 else ('#888' if r == 0 else '#d62728' if r == 1 else '#1f77b4'), zorder=3)
    ax[0].annotate(CARD[r], (pc[r, 0], pc[r, 1]), xytext=(5, 4), textcoords='offset points', fontsize=10)
ax[0].plot(pc[3:15, 0], pc[3:15, 1], color='#bbb', lw=1, zorder=1)
ax[0].set_title(f'牌的向量：前两个主成分（解释 {var[0]:.0%} + {var[1]:.0%}）')
im = ax[1].imshow(C[:15, :15], cmap='RdBu_r', vmin=-1, vmax=1); ax[1].set_xticks(range(15), CARD[:15], rotation=90); ax[1].set_yticks(range(15), CARD[:15])
ax[1].set_title('牌与牌的余弦相似度'); fig.colorbar(im, ax=ax[1], fraction=0.046)
Cp = cos(P); ax[2].imshow(Cp, cmap='RdBu_r', vmin=-1, vmax=1); ax[2].set_xticks(range(8), POSN, rotation=90); ax[2].set_yticks(range(8), POSN)
for i in range(8):
    for j in range(8): ax[2].text(j, i, f'{Cp[i, j]:.2f}', ha='center', va='center', fontsize=7)
ax[2].set_title('8 个位置向量的相似度（列 × 边行/中间行）')
fig.savefig(L / 'fig' / 'embed.png'); plt.close(fig)

fig, ax = plt.subplots(4, 4, figsize=(15, 11.5)); fig.subplots_adjust(hspace=0.5)
lim = np.abs(bias).max()
for l in range(4):
    for h in range(4):
        A = ax[l, h]; A.imshow(bias[l, h], cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
        for i in range(4):
            for j in range(7): A.text(j, i, f'{bias[l, h, i, j]:.1f}', ha='center', va='center', fontsize=7)
        A.set_xticks(range(7), [f'{dc:+d}' for dc in range(-3, 4)]); A.set_yticks(range(4), range(4))
        A.set_title(f'层{l + 1} 头{h + 1}', fontsize=10)
        if h == 0: A.set_ylabel('行差 |Δ行|')
        if l == 3: A.set_xlabel('列差（+ = 被看的格子在左边）')
fig.suptitle('注意力偏置表：格子 → 格子，按相对位置加到注意力分数上（红 = 多看，蓝 = 少看；每个视角里"往左推"）')
fig.savefig(L / 'fig' / 'bias.png'); plt.close(fig)
