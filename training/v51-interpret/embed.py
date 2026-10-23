"""V51 step 1: the input embeddings of the DAgger-14 transformer.
Cards: 16 rank vectors (0 = empty .. 15 = 12288); principal components and cosine similarities.
Cells: 16 learned position vectors; cosine similarities, grouped by corner / edge / centre, and a
check of the 8-fold symmetry (training used a random rotation / mirror per sample).
How often each rank occurs in the training data is reported too (rarely seen ranks keep noisy vectors).
  -> embed.json, fig/embed-cards.png, fig/embed-cells.png"""
import json
import numpy as np
import torch
from common import load_net, CARD, SYM, plt_setup, L, DATA

net = load_net(); E = net.cell.weight.detach().cpu().numpy(); P = net.pos.detach().cpu().numpy()
d = np.load(DATA / 'positions.npz'); freq = np.bincount(d['board'].ravel(), minlength=16) / d['board'].size
cos = lambda X: (X / np.linalg.norm(X, axis=1, keepdims=True)) @ (X / np.linalg.norm(X, axis=1, keepdims=True)).T
Ec = E - E.mean(0); U, S, Vt = np.linalg.svd(Ec, full_matrices=False); pc = Ec @ Vt[:3].T; var = S ** 2 / (S ** 2).sum()
big = list(range(3, 15))                                     # 3 .. 6144: the cards that merge with themselves
r1 = float(np.corrcoef(pc[big, 0], big)[0, 1]); r2 = float(np.corrcoef(pc[big, 1], big)[0, 1])
C = cos(E); Cp = cos(P)
kind = ['corner' if i in (0, 3, 12, 15) else 'centre' if i in (5, 6, 9, 10) else 'edge' for i in range(16)]
groups = {f'{a}-{b}': float(np.mean([Cp[i, j] for i in range(16) for j in range(16) if i != j and kind[i] == a and kind[j] == b]))
          for a, b in (('corner', 'corner'), ('edge', 'edge'), ('centre', 'centre'), ('corner', 'edge'), ('corner', 'centre'), ('edge', 'centre'))}
sym = np.mean([Cp[np.ix_(c, c)] for c, _ in SYM], 0)          # similarity averaged over the 8 symmetries
asym = float(np.abs(Cp - sym)[~np.eye(16, dtype=bool)].mean())
norm = np.linalg.norm(P, axis=1)
dist = np.array([[abs(i // 4 - j // 4) + abs(i % 4 - j % 4) for j in range(16)] for i in range(16)]); off = ~np.eye(16, dtype=bool)
cosDist = float(np.corrcoef(Cp[off], dist[off])[0, 1]); byDist = {int(k): round(float(Cp[off & (dist == k)].mean()), 3) for k in range(1, 7)}
Pc = P - P.mean(0); _, Sp, Vp = np.linalg.svd(Pc, full_matrices=False); pp = Pc @ Vp[:2].T; pvar = Sp ** 2 / (Sp ** 2).sum()
out = {'cardVarianceTop3': var[:3].round(3).tolist(), 'pc1VsRank(3..6144)': round(r1, 3), 'pc2VsRank': round(r2, 3),
       'cosNeighbourRanks(3..6144)': [round(float(C[r, r + 1]), 3) for r in range(3, 14)],
       'cos1_2': round(float(C[1, 2]), 3), 'cos1_3': round(float(C[1, 3]), 3), 'cos2_3': round(float(C[2, 3]), 3), 'cosEmpty_1': round(float(C[0, 1]), 3),
       'rankFrequency': {CARD[r]: float(freq[r]) for r in range(16)}, 'cellGroups': groups, 'cellAsymmetry': round(asym, 3),
       'cellNorm': norm.round(2).tolist(), 'cellCosVsDistance': round(cosDist, 3), 'cellCosByDistance': byDist, 'cellVarianceTop2': pvar[:2].round(3).tolist(), 'cardNorm': np.linalg.norm(E, axis=1).round(2).tolist()}
(L / 'embed.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n'); print(json.dumps(out, ensure_ascii=False))

plt = plt_setup()
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
cm = plt.get_cmap('viridis')
for r in range(15):
    ax[0].scatter(pc[r, 0], pc[r, 1], s=60, color=cm(r / 14) if r >= 3 else ('#888' if r == 0 else '#d62728' if r == 1 else '#1f77b4'), zorder=3)
    ax[0].annotate(CARD[r], (pc[r, 0], pc[r, 1]), xytext=(5, 4), textcoords='offset points', fontsize=10)
ax[0].plot(pc[big, 0], pc[big, 1], color='#bbb', lw=1, zorder=1)
ax[0].set_title(f'牌的向量：前两个主成分（解释 {var[0]:.0%} + {var[1]:.0%}）'); ax[0].set_xlabel('主成分 1'); ax[0].set_ylabel('主成分 2')
im = ax[1].imshow(C[:15, :15], cmap='RdBu_r', vmin=-1, vmax=1); ax[1].set_xticks(range(15), CARD[:15], rotation=90); ax[1].set_yticks(range(15), CARD[:15])
ax[1].set_title('牌与牌的余弦相似度'); fig.colorbar(im, ax=ax[1], fraction=0.046)
fig.savefig(L / 'fig' / 'embed-cards.png'); plt.close(fig)

fig, ax = plt.subplots(1, 5, figsize=(16, 3.4))
for k, ref in enumerate((0, 1, 5, 3, 15)):
    ax[k].imshow(Cp[ref].reshape(4, 4), cmap='RdBu_r', vmin=-1, vmax=1)
    for i in range(16): ax[k].text(i % 4, i // 4, f'{Cp[ref, i]:.2f}', ha='center', va='center', fontsize=8)
    ax[k].set_title(f'与格子 {ref}（{"角" if kind[ref] == "corner" else "中间" if kind[ref] == "centre" else "边"}）的相似度'); ax[k].set_xticks([]); ax[k].set_yticks([])
fig.savefig(L / 'fig' / 'embed-cells.png'); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 5))
for i in range(16):
    for j in (i + 1 if i % 4 < 3 else None, i + 4 if i < 12 else None):
        if j is not None: ax.plot(pp[[i, j], 0], pp[[i, j], 1], color='#bbb', lw=1, zorder=1)
    ax.scatter(*pp[i], s=70, color={'corner': '#d62728', 'edge': '#1f77b4', 'centre': '#2ca02c'}[kind[i]], zorder=3)
    ax.annotate(str(i), pp[i], xytext=(5, 4), textcoords='offset points')
ax.set_title(f'格子位置向量的前两个主成分（解释 {pvar[0]:.0%} + {pvar[1]:.0%}）\n灰线连接棋盘上相邻的格子；红=角，蓝=边，绿=中间', fontsize=10)
ax.set_aspect('equal'); fig.savefig(L / 'fig' / 'embed-cellmap.png'); plt.close(fig)
