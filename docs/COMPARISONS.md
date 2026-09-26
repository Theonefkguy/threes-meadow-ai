# Public comparisons / 公开结果对照

Checked 2026-09-26. These are source-reported results under different setups, not a common leaderboard. We do not claim an established global ranking.

| Work | Reported 6144 rate | Context |
|---|---:|---|
| This project V24 | 39.1% (400/1024) | Modern community-reconstructed rules, candidate preview, five-ply cutoff search, fresh standard openings |
| [Josiah Kiok, 2026](https://medium.com/@josiah-kiok/beating-threes-with-reinforcement-learning-ae074dd28a68) | 21.5% (10,000 games) | PPO without search in its own simulator; information and compute budgets differ |
| [Yeh et al., 2016](https://arxiv.org/abs/1606.07374) | 7.83% | Historical paper rules and MS-TD setup; not a modern-rule rerun |
| [nneonneo/threes-ai](https://github.com/nneonneo/threes-ai) | No formal rate in README | README reports reaching 6144; no comparable denominator |

Kiok's 14%–20% discussion refers to **halfrost**, not nneonneo. Kiok's approximately 0.065% 12288 figure describes training experience, not the same formal 10,000-game evaluation. Do not present either as a controlled head-to-head result.

中文：本项目的 39.1% 高于上表中已列出的公开报告数字，但规则、预告信息、搜索预算和评测过程不同，不能由此认定统一条件下的「公开最强」。若要建立该结论，应固定对手版本，在相同规则、可见信息和明确计算预算下重新评测，并公开原始数据。本次核对覆盖上述来源，不是对全部公开项目的穷尽检索。
