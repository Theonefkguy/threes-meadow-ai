# Reproduction / 复现

Run all commands from the repository root. Requires Python 3.9+, Node.js 22 for JS tests and a C++17 compiler. These V24 commands do not require external Python packages.

## Verify the released evidence / 校验发布证据

```sh
node --test tests/*.test.mjs
python3 scripts/verify-release.py
```

This checks model hashes, unique expected seeds, aggregate counts and re-analysis of archived records in a temporary directory. It leaves the original records unchanged. / 校验模型哈希、种子完整性、里程碑和重算统计，原始数据不变。

## Replay a sample / 重跑抽样局

```sh
mkdir -p work
g++ -std=c++17 -O3 training/v24-acceptance/game.cpp -o work/v24-game
python3 scripts/verify-release.py --replay work/v24-game
```

This reruns seed 24110001 for both arms in temporary files and compares every recorded field except CPU timings. / 重跑新旧两组的首个验收种子，除耗时外逐字段对照记录。

## Full 2 × 1,024 games / 全量整局验收

```sh
g++ -std=c++17 -O3 -march=native training/v24-acceptance/game.cpp -o /tmp/v24-game
python3 training/v24-acceptance/run.py
python3 training/v24-acceptance/analyze.py
```

Use a separate clone for a full rerun: these original scripts write `training/v24-acceptance/games/`, `games-OLD.jsonl`, `games-NEW.jsonl` and `results.json`. Completed 32-game chunks are reused on resume. Do not reuse chunks after changing model weights or engine/search settings. Full runs can take hours depending on hardware; there is no fixed ETA. `-march=native` is optional and machine-specific.

建议在独立克隆中全量运行：原脚本会写入分片和最终记录，并复用已完成的 32 局分片。修改模型或引擎设置后不可复用旧分片。耗时随硬件变化，可能需要数小时。

NEW: V23 weights, depth 5, cutoff 0.01, no node cap, leaf clamp 0. OLD: V4 weights, depth 3, 24,000-node budget, no cutoff or clamp. Standard starts use seeds 24110001–24111024. The native NEW run has no browser wall-time budget. After policies diverge, later random events are not coupled one-to-one.

## Training / 训练

The original protocols and reproduction recipes are retained in [V21](../RL-V21-LATE-TRAIN.md), [V23](../RL-V23-STAGE2.md), and earlier round reports. `training/v7/base.ntd` is included (SHA-256 `d863b01e…`; formerly also published as `dist/models/ntuple-v8.bin`). The V21 website model is kept at `training/v21-late-train/models/ntuple-v21.bin`.

The slim release removed the V1–V3 and V5–V8 website models and their strategies, and purged those weight files from git history. Scripts of those earlier rounds that load `dist/models/ntuple-v1.bin`…`ntuple-v8.bin` no longer run from this repository; only `dist/models/ntuple-v4.bin` and `ntuple-v23.bin` are published. Historical runs through V19 used legacy bonus rules; set `THREES_BONUS_RULE=legacy` where required by their protocols, not for V24.

These are archived research workflows, not a single deterministic end-to-end training command. Some scripts depend on earlier generated pools and stop by CPU time. Cross-platform bit-identical retraining has not been verified. V24 evaluates the released frozen weights, whose SHA-256 values are in its protocol.

训练代码和实验记录保留完整，但历史流程存在前置局面库及按 CPU 时间停止的实验，尚未验证跨平台从零重训的逐位一致性。V24 复现使用的是随仓库发布的固定权重。

## Release validation / 发布验证

2026-09-26: archive CRC check passed; 48 original tests passed; C++ evaluation compiled; both model hashes matched; analysis reproduced (only negligible float rounding in timing statistics); the first NEW and OLD game matched all non-timing fields. Full 2,048-game rerun and full retraining were not performed for release preparation.
