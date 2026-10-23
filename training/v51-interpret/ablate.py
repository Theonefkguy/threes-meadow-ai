"""V51 step 4: what each part does, by removing it.
Mean ablation: one head's output (or one layer's MLP output) replaced by its mean over 8192 validation
positions. Measured on the 30000 sampled validation positions (near / bad5 vs the AI) and in 300 whole
games with no search (seeds 50900001.., the V50 play.py games, mean score and reach rates).
Occlusion maps: in 6 late-game positions where the model picks the AI's best move clearly (second-best
direction >= 1% worse), each cell is set to every other possible value (empty, 1, 2, 3, 6 .. the largest
card); the map shows the average drop of the probability of the chosen direction.
  -> ablate.json, fig/ablate.png, fig/occlusion.png"""
import json, sys, time
import numpy as np
import torch
from common import load_net, sample, to_dev, run, mean_outputs, score, gaps, plt_setup, L, V50, CARD, dev
from engine import Game, move

net = load_net(); a = sample(); M = mean_outputs(net, a)
def evaluate(abl):
    logits = torch.cat([run(net, to_dev(a, slice(i, i + 4096)), ablate=abl, means=M)[0] for i in range(0, len(a['board']), 4096)])
    return score(logits, a['values'])
def play(abl, n=300):
    games = [Game(50900001 + i) for i in range(n)]; alive = [g for g in games if not g.over]
    while alive:
        x = {'board': [g.b for g in alive], 'cards': [(g.next['cards'] + [0, 0, 0])[:3] for g in alive], 'bonus': [g.next['bonus'] for g in alive],
             'counts': [g.counts for g in alive], 'cycle': [list(g.cycle) for g in alive], 'legal': [[bool(move(g.b, d)[1]) for d in range(4)] for g in alive]}
        x = to_dev({k: np.array(v) for k, v in x.items()})
        for g, d in zip(alive, run(net, x, ablate=abl, means=M)[0].argmax(1).tolist()): g.step(d)
        alive = [g for g in alive if not g.over and g.turns < 50000]
    top = [max(g.b) for g in games]
    return {'meanScore': float(np.mean([g.score() for g in games])), 'meanMoves': float(np.mean([g.turns for g in games])),
            'reach1536': float(np.mean([t >= 12 for t in top])), 'reach3072': float(np.mean([t >= 13 for t in top]))}
if __name__ == '__main__':
    parts = [('none',)] + [('head', l, h) for l in range(4) for h in range(4)] + [('mlp', l) for l in range(4)]
    res = []
    for p in parts:
        t0 = time.time(); abl = [] if p == ('none',) else [p]
        r = {'part': p, **evaluate(abl), **play(abl)}; res.append(r)
        print(p, {k: round(v, 4) for k, v in r.items() if k != 'part'}, f'{time.time() - t0:.0f}s', flush=True)
    # occlusion maps
    v = a['values']; g = gaps(v); x = to_dev(a); logits = torch.cat([run(net, to_dev(a, slice(i, i + 4096)))[0] for i in range(0, len(v), 4096)])
    pick = logits.argmax(1).cpu().numpy(); gs = np.sort(g, 1)
    cand = np.flatnonzero((g[np.arange(len(pick)), pick] == 0) & (gs[:, 1] >= 0.01) & (a['phase'] >= 12))
    ex = np.random.default_rng(3).choice(cand, 6, replace=False); maps = []
    for i in ex:
        b0 = a['board'][i].astype(int); top = b0.max(); d0 = int(pick[i])
        def prob(boards):
            n = len(boards); y = {k: np.repeat(a[k][i:i + 1], n, 0) for k in ('cards', 'bonus', 'counts', 'cycle')}
            y['board'] = np.array(boards, np.uint8); y['legal'] = np.array([[bool(move(list(bb), d)[1]) for d in range(4)] for bb in boards])
            return torch.softmax(run(net, to_dev(y))[0], 1)[:, d0].cpu().numpy()
        p0 = prob([b0])[0]; m = np.zeros(16)
        for c in range(16):
            alts = [r for r in range(top + 1) if r != b0[c]]; boards = []
            for r in alts: bb = b0.copy(); bb[c] = r; boards.append(bb)
            m[c] = float(np.mean(p0 - prob(boards)))
        maps.append({'index': int(i), 'board': b0.tolist(), 'cards': a['cards'][i].tolist(), 'dir': d0, 'p': float(p0), 'drop': m.tolist(),
                     'gaps': [None if np.isinf(z) else float(z) for z in g[i]]})
    (L / 'ablate.json').write_text(json.dumps({'ablation': res, 'occlusion': maps}, ensure_ascii=False, indent=1) + '\n')

    plt = plt_setup()
    base = res[0]; lab = ['无'] + [f'层{l + 1}头{h + 1}' for l in range(4) for h in range(4)] + [f'层{l + 1} MLP' for l in range(4)]
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True)
    y = np.arange(len(res))[::-1]
    ax[0].barh(y, [r['near'] * 100 for r in res], color=['#555'] + ['#4c78a8'] * 16 + ['#e45756'] * 4)
    ax[0].axvline(base['near'] * 100, color='k', lw=0.8, ls='--'); ax[0].set_yticks(y, lab); ax[0].set_xlabel('与 AI 接近的比例 (%)'); ax[0].set_xlim(30, 85)
    ax[1].barh(y, [r['reach3072'] * 100 for r in res], color=['#555'] + ['#4c78a8'] * 16 + ['#e45756'] * 4)
    ax[1].axvline(base['reach3072'] * 100, color='k', lw=0.8, ls='--'); ax[1].set_xlabel('不带搜索 300 局合出 3072 的比例 (%)')
    for k, r in enumerate(res):
        ax[0].text(r['near'] * 100 + 0.3, y[k], f"{r['near'] * 100:.1f}", va='center', fontsize=8)
        ax[1].text(r['reach3072'] * 100 + 0.3, y[k], f"{r['reach3072'] * 100:.0f}", va='center', fontsize=8)
    fig.suptitle('均值消融：把一个头（蓝）或一层 MLP（红）换成平均输出'); fig.savefig(L / 'fig' / 'ablate.png'); plt.close(fig)

    DIR = ['←', '→', '↑', '↓']
    fig, ax = plt.subplots(2, 3, figsize=(13, 9))
    for k, mp in enumerate(maps):
        A = ax.flat[k]; dm = np.array(mp['drop']).reshape(4, 4)
        A.imshow(dm, cmap='Reds', vmin=0, vmax=max(0.3, dm.max()))
        for c in range(16): A.text(c % 4, c // 4, f"{CARD[mp['board'][c]] if mp['board'][c] else '·'}\n{mp['drop'][c]:.2f}", ha='center', va='center', fontsize=10)
        cards = '/'.join(CARD[c] for c in mp['cards'] if c)
        gp = ' '.join(f"{DIR[d]}{'—' if z is None else f'{z * 100:.1f}%'}" for d, z in enumerate(mp['gaps']))
        A.set_title(f"选 {DIR[mp['dir']]}（概率 {mp['p']:.2f}），提示牌 {cards}\nAI 各方向差距：{gp}", fontsize=9); A.set_xticks([]); A.set_yticks([])
    fig.suptitle('逐格改变牌值，所选方向的概率平均下降多少（越红越关键）'); fig.savefig(L / 'fig' / 'occlusion.png'); plt.close(fig)
