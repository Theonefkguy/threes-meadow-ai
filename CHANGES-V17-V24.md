# 合三 AI：V17–V24 改动索引

| 轮次 | 报告 | 代码 / 数据 |
|---|---|---|
| V17 概率剪枝搜索 | RL-V17-FAST-SEARCH.md | training/v17-fast-depth, dist/fast-search.js |
| V18 深搜到 6144（旧规则） | RL-V18-LATE.md | training/v18-late |
| V19 从 3072 局面比较更深搜索 | RL-V19-DEPTH-SNAPSHOTS.md | training/v19-snapshots |
| V20 官方现代规则 + 12288 结局 | RULES.md | training/v20-rules, training/v3/ntuple-fast.h, dist/engine.js |
| V21 后期重训 + 第 3 段（6144+） | RL-V21-LATE-TRAIN.md | training/v21-late-train, dist/models/ntuple-v21.bin |
| V22 叶节点下限 0 | RL-V22-CLAMP.md | training/v22-clamp |
| V23 第 2 段改进训练（当前网页模型） | RL-V23-STAGE2.md | training/v23-stage2, dist/models/ntuple-v23.bin |
| V24 整局总验收 | RL-V24-ACCEPTANCE.md | training/v24-acceptance |

网页入口：dist/index.html（静态文件，需通过 HTTP 服务打开，例如在 dist 目录运行 `python3 -m http.server`）。
默认 AI：五层剪枝搜索（阈值 0.01，叶节点下限 0），模型 ntuple-v23.bin。
测试：`node --test tests/*.test.mjs`（48 项）。
C++ 实验需要 `training/v7/base.ntd`（与 dist/models/ntuple-v8.bin 相同，已包含）。
旧规则复现：设置环境变量 `THREES_BONUS_RULE=legacy`。
