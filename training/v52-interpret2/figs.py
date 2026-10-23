"""V52 figures from simulate.json, sae.json / sae-top.json, dla.json.  -> fig/*.png"""
import json, sys
from pathlib import Path
import numpy as np
L = Path(__file__).resolve().parent; sys.path.insert(0, str(L.parent / 'v51-interpret'))
import common
from common import sample, CARD
common.FIG = L / 'fig'; plt = common.plt_setup()

s = json.load(open(L / 'simulate.json'))['average']; kinds = ['不变', '被推进来', '合并点', '变空']
fig, ax = plt.subplots(figsize=(7, 4.2)); cols = ['#888', '#4c78a8', '#e45756', '#54a24b']
for k, c in zip(kinds, cols):
    ax.plot(range(5), [x[k] * 100 for x in s['layers']], marker='o', color=c, label=f'{k}（占 {s["share"][k]:.0%}）')
    ax.axhline(s['baseline'][k] * 100, color=c, ls=':', lw=1)
ax.set_xticks(range(5), ['输入', '第1层', '第2层', '第3层', '第4层']); ax.set_ylabel('识别率 (%)'); ax.set_ylim(-3, 103)
ax.set_title('格子 token 能否读出"往某方向推后，这一格会怎样"\n（实线：各层；点线：直接用输入的线性基线）', fontsize=10); ax.legend(fontsize=9, loc='lower right')
fig.savefig(L / 'fig' / 'simulate.png'); plt.close(fig)

d = json.load(open(L / 'dla.json')); names = d['components']; keep = [n for n in names if n not in ('嵌入', '偏置')]
fig, ax = plt.subplots(figsize=(8, 6)); y = np.arange(len(keep))[::-1]
ax.barh(y + 0.2, [d['decision'][n] for n in keep], 0.4, color='#4c78a8', label='平时：推向所选方向的量')
ax.barh(y - 0.2, [d['towardWrong'][n] for n in keep], 0.4, color='#e45756', label=f'大错局面（{d["bad5"]} 个）：推向错误方向、远离 AI 首选的量')
ax.set_yticks(y, keep); ax.axvline(0, color='k', lw=0.6); ax.legend(fontsize=9, loc='lower right'); ax.set_xlabel('对方向 logit 的直接贡献')
ax.set_title('谁在直接写入最后的决定', fontsize=11); fig.savefig(L / 'fig' / 'dla.png'); plt.close(fig)

a = sample(); sae = json.load(open(L / 'sae.json'))['sites']; top = json.load(open(L / 'sae-top.json'))
pick = [('final', 942, '"往 ↑ 推会动到大牌"'), ('final', 9, '"往 ← 推会动到大牌"'), ('L2', 850, '"提示的是奖励牌"'), ('final', 439, '"往 → 推就要卡死"'),
        ('final', 100, '"AI 估分极低，\n多个方向不安全"'), ('L2', 871, '"最大牌在右下角"')]
fig, ax = plt.subplots(len(pick), 4, figsize=(10, 2.7 * len(pick)))
for r, (site, f, title) in enumerate(pick):
    info = next(x for x in sae[site]['features'] if x['id'] == f)
    for c, i in enumerate(top[site][str(f)][:4]):
        A = ax[r, c]; b = a['board'][i]; A.imshow(np.log2(np.maximum(b, 1)).reshape(4, 4), cmap='YlOrBr', vmin=0, vmax=4)
        for k in range(16): A.text(k % 4, k // 4, CARD[b[k]] if b[k] else '', ha='center', va='center', fontsize=9)
        nx = '/'.join(CARD[x] for x in a['cards'][i] if x) + ('（奖励）' if a['bonus'][i] else '')
        A.set_xticks([]); A.set_yticks([]); A.set_xlabel(f'提示 {nx}', fontsize=8)
    ax[r, 0].set_ylabel(f'{site} #{f}\n{title}\n频率 {info["freq"]:.1%}', fontsize=9, rotation=0, ha='right', va='center')
fig.suptitle('稀疏自编码器特征：激活最强的 4 个局面', fontsize=12); fig.tight_layout(rect=[0, 0, 1, 0.98]); fig.savefig(L / 'fig' / 'sae-examples.png'); plt.close(fig)
