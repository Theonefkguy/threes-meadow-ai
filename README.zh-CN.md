# 合三 · Threes Meadow AI

[在线试玩](https://theonefkguy.github.io/threes-meadow-ai/) · [English](README.md) | **简体中文**

一个可以在浏览器里观看、随时接管的高性能 Threes 游戏 AI。结合多阶段时序差分学习（4 格与 5 格 N-tuple 查表）与五层期望搜索，使用概率剪枝控制计算量。模型完全在你的设备上运行，无需账号、API 密钥或推理服务器。

**6144 达成率 59.3% · 12288 达成率 5.6% · 2048 局全新开局测试。**

## 试玩

打开[游戏页面](https://theonefkguy.github.io/threes-meadow-ai/)，选择中文或 English，点击「AI 自动玩」。使用方向键、滑动或方向按钮即可手动接管。默认 V31 模型约 32 MB，首次使用时下载并显示进度条；也可切换到第四版或经典搜索进行对比。

## 视频介绍

[![How a Threes AI Actually Thinks](https://i.ytimg.com/vi/Rov_8bpDScY/hqdefault.jpg)](https://www.youtube.com/watch?v=Rov_8bpDScY)

在 YouTube 观看 **How a Threes AI Actually Thinks（Threes AI 如何思考）**。

## V31 发布成绩

规则和引擎与下文相同。用原生 C++ 评测实际发布到网页的权重（16 位浮点残差），从标准开局玩到 12288 或死亡，两组使用相同的 2048 个开局种子。

| 指标 | 原网页 AI（V23） | V31（发布版） |
|---|---:|---:|
| 合出 3072 | 87.5% | 92.9% |
| 合出 6144 | 790/2048（38.6%） | **1215/2048（59.3%）** |
| 合出 12288 | 79/2048（3.9%） | **115/2048（5.6%）** |
| 平均最终得分 | 438,894 | **561,463** |

V31 的 95% Wilson 区间：6144 为 **57.2%–61.4%**；12288 为 **4.7%–6.7%**。三个主要指标经 Holm 校正后均显著（6144 p=10⁻⁴⁴，12288 p=0.009，得分 p=3×10⁻³³）。12288 的提升来自更多对局走到了 6144：一旦合出 6144，两个 AI 之后合出 12288 的比例都约为 10%（V23 79/790，V31 115/1215）。到目前为止，针对第 3 段的实验（V27、V31、[V32](RL-V32-STRONG.md)）都没能提高这个比例。提升来自第 1–3 段新增的 5 格残差表（V29–V31）：只继续训练 4 格表没有提升，加大模式后才有。320 个抽样局面上 JS 与 C++ 的走法完全一致；桌面 Chromium 平均每步 5.4 ms（P95 9.0 ms）。

[V31 报告](RL-V31-STAGES.md) · [发布检查](training/web-release/v31-check/) · [V29 5 格测试](RL-V29-FIVE.md) · [V30 并行训练](RL-V30-PARALLEL.md)

## V24 整局验收（上一版）

遵循社区逆向确认的现代 Threes 核心规则，采用标准九张牌开局。使用原生 C++，从开局玩到合出 12288 或死亡，新旧两组使用相同的 1024 个开局种子。

| 指标 | 项目改进前 AI | V23 模型 + 五层搜索 |
|---|---:|---:|
| 合出 1536 | 93.8% | 98.8% |
| 合出 3072 | 65.4% | 86.2% |
| 合出 6144 | 114/1024（11.1%） | **400/1024（39.1%）** |
| 合出 12288 | 0/1024 | **31/1024（3.0%）** |
| 平均最终得分 | 227,400 | **432,649** |

95% Wilson 区间：6144 为 **36.1%–42.1%**；12288 为 **2.1%–4.3%**。相对于项目旧 AI 的三个主要指标经 Holm 校正后均显著。此表是项目内部基线比较，不是对全部公开 AI 的统一排名。

[复现说明](docs/REPRODUCING.md) · [V24 报告](RL-V24-ACCEPTANCE.md) · [评测协议](training/v24-acceptance/protocol.json) · [逐局记录](training/v24-acceptance/games-NEW.jsonl) · [公开结果对照](docs/COMPARISONS.md)

网页有 160 ms 搜索软预算；V24 原生 NEW 组不设这个时间上限，每步完整完成五层。较慢设备可能只完成较浅搜索。成绩来自本仓库引擎，尚未在官方 App 上实测。部分训练实验按 CPU 时间停止，不同硬件不保证重训得到相同权重；仓库保留被评测的权重和哈希。

## 本地运行

需要 Python 3 提供静态服务；测试建议使用 Node.js 22。无需 npm 安装或构建步骤。

```sh
git clone https://github.com/Theonefkguy/threes-meadow-ai.git
cd threes-meadow-ai
python3 -m http.server 8000 --bind 127.0.0.1 --directory dist
```

访问 http://localhost:8000。请通过 HTTP 服务打开，避免直接双击 HTML 导致模块 Worker 和模型加载失败。

```sh
node --test tests/*.test.mjs
python3 scripts/verify-release.py
```

## 工程结构

- `dist/`：可直接编辑的网页代码、样式和模型。
- `training/`：训练与评测代码、协议、历史实验数据。
- `tests/`：规则、模型、搜索、输入与 AI 生命周期测试。
- `docs/`：复现、对照和发布说明。
- `.github/workflows/pages.yml`：先测试、校验，再仅发布 `dist/`。

当前模型（V31）第 0 段沿用 V4；第 1–3 段（最大牌 1536、3072、6144 以上）为 4 格表加 5 格残差表，用并行 TD 学习训练（V30、V31）。AI 只使用棋盘、预告候选及通过公开历史推断的剩余牌数，不读取隐藏牌序。[算法历史](AI.md)与[规则说明](RULES.md)保留在仓库中。

## 许可

代码及本项目模型权重采用 [MIT](LICENSE)。[来源说明](NOTICE.md)。本项目是独立研究实现，与 Threes! / Sirvo 无隶属关系或背书关系。
