# 合三强化学习第二版

第二版以提高正常开局合出 3072 的比例为主要评测目标。实际结果见 [RL-V2-RESULTS.md](RL-V2-RESULTS.md)。游戏规则、发牌概率、公开信息边界沿用第一版。训练和独立评测均实际运行；模型和最终结果由 `training/v2/selection.json`、`training/v2/evaluation.json` 记录。

## 候选方法

A：从 V1 的权重继续训练 400,000 轮，正常开局和真实的 1536 中局各 200,000 轮。学习目标仍是未来得分，但从 TD(0) 改为五步截断的 TD(λ)，λ=.5，回报权重为 .5、.25、.125、.0625、.0625。根据 afterstate 实际最大牌切换阶段。到达 3072 后以冻结的 V1 后期估值作边界，集中更新前、中期权重。

中局来自 V1 保存的实际对局，随后补充本次训练新到达的 1536 局面。每个池最多 5,000 条蓄水池样本。中局抽样有 20% 概率使用临近失败前的真实状态；没有直接在随机棋盘上放置大牌。多步目标使用整段轨迹更新前的预测值。原有对称特征重复命中的平方范数归一化仍然保留。

B：额外训练一个预测达到 3072 的 sigmoid 概率头，使用相同的四格图案和棋盘对称性。先用 V1 得分策略生成 20,000 轮实际结局作 Monte Carlo 热身，再用五步 TD(λ) 训练。

C：在 B 的概率头中加入公开的来牌候选、普通牌剩余计数、空格数、1/2 数量及两种小牌的剩余差。两张额外特征表使用确定性无碰撞索引。预告权重仍由搜索中的游戏规则处理，概率头的特征不包含隐藏牌组顺序。

纯概率策略的初次试验出现严重退化，日志保留在 run-b / run-c，源码保留在 train-goal-pure.cpp / goal-pure.h。后续 B/C 使用「V1 得分价值 + 60,000 × 学到的成功率」作辅助决策，学习率从 .2 降为 .04。60,000 是固定试验参数，没有用独立测试集调节。这个组合是决策效用，不能向玩家展示成校准后的真实胜率。

A 完成 400,000 轮；B/C 先以 100,000 轮作筛选，如果没有同时超过 V1 与完成训练的 A，就停止该方向。候选训练预算、目标和超参数不同，这不是严格的单变量消融试验。被放弃的试验不会算入获选模型的训练轮数。

## 公开信息与边界

- 策略只接收棋盘、当前候选预告和公开观察得到的普通牌剩余计数；网页不会将牌组或随机数发生器传给 Worker。
- 从快照恢复公开计数时只恢复剩余张数。隐藏的排列只供环境继续发牌，不供策略查看。
- V1 快照中的概率曾保留六位小数；加载中局时根据公开候选窗口重建精确条件概率。
- 成功事件为首次出现任意 >=3072 的牌；达成后网页继续玩，使用得分策略。
- 概率头的终局失败为 0、达成为 1。A 的得分头在终局仍计入最后一张来牌分数，3072 边界则保留后续得分估值。

## 评测设计

先用种子 910001–910200，从正常九牌开局对 V1 和候选模型各评测 200 局。选择初筛成功次数最多的候选，平手优先 A、B、C。模型校验和固定后，再比较种子 1910001–1911000 的 1,000 局独立测试。V1 的独立测试可提前计算，但直到候选固定才读取结果，避免用该集合选模型。

每局在达到目标或无路可走时结束。6,000 步未完成的局面单独标记截断；不能算成功，也不能伪装成终局。因此记录的 `scoreAtStop` 不是完整终局分数，报告不拿它比较平均终局成绩。

C++ 评测沿用网页的三步期望搜索、24,000 节点上限、迭代加深和只采用已完成深度的策略。它不设墙钟超时，以消除机器负载噪声。网页另有 160 ms 软预算，低性能设备可能搜索更浅；离线达成率并非手机上的保证。

C++ / JavaScript 已在 690 个 V1 局面、217 个带公开特征的辅助目标局面上逐一核对方向、完成深度及节点数。环境与实际 JS 游戏规则的一致性沿用并重跑 V1 的引擎验证。测试覆盖多步回报、终局和成功边界、概率表格式、公开特征、Worker 模型加载和失败回退，以及自动玩的停止与手动接管。

最终统计包含 Wilson 95% 区间、配对差值近似 95% 区间和双侧精确 McNemar 检验。同一种子保证开局相同，策略分岔后的来牌并不逐张相同。选择模型只使用初筛集合；独立测试结果不会用于调参或换模型。

## 复现

使用 C++17 和近期 Node.js，在项目根目录执行：

```sh
mkdir -p training/v2/bin
python training/v2/prepare-curriculum.py
g++ -std=c++17 -O3 -march=native training/v2/train-score.cpp -o training/v2/bin/train-score
g++ -std=c++17 -O3 -march=native training/v2/train-goal.cpp -o training/v2/bin/train-goal
g++ -std=c++17 -O3 -march=native training/v2/benchmark.cpp -o training/v2/bin/benchmark
training/v2/bin/train-score training/v2/reproduce-a 400000 202609241 dist/models/ntuple-v1.bin training/v2/initial-curriculum.txt
training/v2/bin/train-goal training/v2/reproduce-b 400000 202609243 dist/models/ntuple-v1.bin training/v2/initial-curriculum.txt 0 20000
training/v2/bin/train-goal training/v2/reproduce-c 400000 202609243 dist/models/ntuple-v1.bin training/v2/initial-curriculum.txt 1 20000
```

B/C 的试验取 `goal-100000.goal` 检查点，使用原定 400,000 轮的学习率日程，因此不能将命令中的总轮数直接改成 100,000 后声称模型相同。完整训练过程和所用检查点轮数分开记录。

```sh
training/v2/bin/benchmark dist/models/ntuple-v1.bin - 910001 200 training/v2/reproduce-screen-v1.jsonl 0
training/v2/bin/benchmark training/v2/reproduce-a/checkpoint.ntd - 910001 200 training/v2/reproduce-screen-a.jsonl 0
training/v2/bin/benchmark dist/models/ntuple-v1.bin training/v2/reproduce-b/goal-100000.goal 910001 200 training/v2/reproduce-screen-b.jsonl 0
training/v2/bin/benchmark dist/models/ntuple-v1.bin training/v2/reproduce-c/goal-100000.goal 910001 200 training/v2/reproduce-screen-c.jsonl 0
node --test tests/*.test.mjs
node training/verify-worker.mjs
```

最终模型文件、校验和、初筛及独立测试原始记录随源码保存。方法参考仍为 [Multi-Stage Temporal Difference Learning for 2048-like Games](https://arxiv.org/abs/1606.07374)。这是轻量独立实现，不是论文全部训练规模和特征的复现；论文的达成率不代表本网站的成绩。
