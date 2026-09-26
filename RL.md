# 合三强化学习第一版

> 此文保留第一版的方法与历史小样本结果。第二版及新的独立评测见 [RL-V2.md](RL-V2.md)。

第一版是独立实现、从零训练的轻量 MS-TD 原型，不是论文完整复现，也没有使用别人的模型权重。

## 学到的内容

- 8 种四格局部图案，每种共享棋盘的 8 种旋转/镜像变换；三个阶段各有 524,288 个 Float32 权重。
- 第一阶段从正常九牌开局自我对局；中期从真正到达 1536 的对局快照开始；后期从真正到达 3072 的快照开始。快照最多保留 5,000 个，采用蓄水池抽样。
- 后续阶段先继承前一阶段的权重，再继续 TD(0) 学习。每一阶段训练到终局，跨过后续阈值不人为终止，也不把过关当作额外奖励。
- 学习对象是移动完成、尚未补入新牌的棋盘（afterstate）。预测未来尚未获得的分数，不包含当前棋盘已有分数。
- TD 目标为「上次移动后新牌带来的分数 + 下一步合并增分 + 下一 afterstate 价值」；终局目标只有最后一张新牌的分数。gamma=1，学习率依训练进程为 .05、.02、.008。
- 对称特征重合时按特征向量平方范数归一化更新，避免相同参数被重复命中时学习率失控。
- 这是仅以棋盘为输入的近似评分器，未直接把预告和剩余牌数编码进评分表。浏览器的期望搜索会单独、精确地使用这些公开信息。训练决策也不读取隐藏的牌袋排列。
- 先用一步策略从零预训练 100 万局，再结合两步搜索做分阶段训练。两步训练会使用公开的来牌提示；随机发牌提供随机轨迹。没有人工评分教师或伪造高阶牌开局。

## 网页使用

棋盘下方默认选择「强化学习 · 第一版」，点击「AI 自动玩」即可。模型约 6 MiB，在 Worker 中加载和计算；选择「经典搜索」可以继续使用原 AI。

期望搜索保留最多三步、24,000 节点和 160 ms 的软预算，仅采用完整算完的深度。评分器处于补牌前，经典 AI 的评分处于补牌后，两者的层数不代表完全相同的计算树；比较以相同计算预算为准。网页支持停止、手动接管、后台自动停止、切换策略中止旧请求。模型加载失败会明确提示并使用经典 AI。

最高牌超过 12,288 时使用经典 AI；该范围之外没有充分训练数据。训练不会让网页的出牌概率发生变化。

## 复现

需要 C++17 编译器和 Node.js，不需要 GPU 或 API 密钥。从项目目录运行：

```sh
mkdir -p training/bin
g++ -std=c++17 -O3 -march=native training/train.cpp -o training/bin/train
g++ -std=c++17 -O3 training/probe.cpp -o training/bin/probe
node training/verify-engine.mjs
training/bin/train training/pretrain-reproduction 1000000 0 0 20260922
training/bin/train training/run-v1-refined 30000 30000 30000 20260923 training/pretrain-reproduction/stage-0.ntd 2
node training/export.mjs training/run-v1-refined
node training/verify-model.mjs
node --test tests/*.test.mjs
node training/compare.mjs dist/models/ntuple-v1.bin 12 810001 training/evaluation-reproduction.json
node training/summarize.mjs training/evaluation-reproduction.json
```

训练输出包含每 5,000 局的日志、阶段模型、最终检查点、随机种子和真实残局快照。最终网页模型及其 SHA-256 在 dist/models 中。对比脚本默认跳过已记录的种子；使用另一个输出文件名进行全新测试。不同策略共用开局种子，不保证决策分岔后的来牌逐张相同。

也可解压 training/pretraining-stage0.ntd.gz，跳过预训练，从保存的准确权重开始第二步。用于初始化的预训练记录在 training/pretraining.json。探索阶段另跑过较小试验和后期训练，未被最终模型使用的权重不计入最终模型训练量。

训练中从中后期开始的最终分数已经包含起点的大牌，不能与正常开局的测试成绩混用。完整测试只从正常九牌开局开始，并记录是否真正到达终局。

## 方法参考

[Multi-Stage Temporal Difference Learning for 2048-like Games](https://arxiv.org/abs/1606.07374) 提供分阶段训练、afterstate 价值和期望搜索的思路。原论文训练规模、特征和实现不同，论文成绩不是本模型成绩。
