# 合三 AI：V17–V24 改动索引

| 轮次 | 报告 | 代码 / 数据 |
|---|---|---|
| V17 概率剪枝搜索 | RL-V17-FAST-SEARCH.md | training/v17-fast-depth, dist/fast-search.js |
| V18 深搜到 6144（旧规则） | RL-V18-LATE.md | training/v18-late |
| V19 从 3072 局面比较更深搜索 | RL-V19-DEPTH-SNAPSHOTS.md | training/v19-snapshots |
| V20 官方现代规则 + 12288 结局 | RULES.md | training/v20-rules, training/v3/ntuple-fast.h, dist/engine.js |
| V21 后期重训 + 第 3 段（6144+） | RL-V21-LATE-TRAIN.md | training/v21-late-train（模型现位于 training/v21-late-train/models/ntuple-v21.bin） |
| V22 叶节点下限 0 | RL-V22-CLAMP.md | training/v22-clamp |
| V23 第 2 段改进训练（当前网页模型） | RL-V23-STAGE2.md | training/v23-stage2, dist/models/ntuple-v23.bin |
| V24 整局总验收 | RL-V24-ACCEPTANCE.md | training/v24-acceptance |
| V25 3072 后降低剪枝阈值（未采用） | RL-V25-ADAPTIVE-CUTOFF.md | training/v25-adaptive-cutoff |
| V26 6144 后加强搜索（未采用） | RL-V26-LATE-CUTOFF.md | training/v26-late-cutoff |
| V27 第 3 段按 V23 方法重训（未采用） | RL-V27-STAGE3.md | training/v27-stage3 |
| V28 V27 训练曲线（探索性） | RL-V28-CURVE.md | training/v28-curve |
| V29 第 2 段加 5 格残差（离线，+13.3 个百分点） | RL-V29-FIVE.md | training/v29-five |
| V30 并行训练残差 + 整局评测（离线，6144：39.7% → 52.7%） | RL-V30-PARALLEL.md | training/v30-parallel |
| V31 第 1、3 段加 5 格残差（离线，12288：4.2% → 5.9%） | RL-V31-STAGES.md | training/v31-stages |
| V32 第 3 段用强搜索对局训练（无效） | RL-V32-STRONG.md | training/v32-strong |
| V33 第 3 段换 5 格模式形状的筛选（停止） | RL-V33-PATTERNS.md | training/v33-patterns |

网页入口：dist/index.html（静态文件，需通过 HTTP 服务打开，例如在 dist 目录运行 `python3 -m http.server`）。
默认 AI：五层剪枝搜索（阈值 0.01，叶节点下限 0），模型 ntuple-v23.bin。
测试：`node --test tests/*.test.mjs`（40 项）。
C++ 实验需要 `training/v7/base.ntd`（即原 dist/models/ntuple-v8.bin，已包含）。
精简版：网页仅保留默认、第四版与经典搜索；第一至三版、第五至八版模型已从仓库及历史中移除。
旧规则复现：设置环境变量 `THREES_BONUS_RULE=legacy`。
