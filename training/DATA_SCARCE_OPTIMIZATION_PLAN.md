# 数据不足阶段的模型优化方案（Data-Scarce Optimization Plan）

> 当前真实人类数据：**386 条轨迹 / 38 局**（`artifacts/human_traces.jsonl`，已去重清洗）。
> 线上部署模型：v3-lite（LayerNormMLP + 动作价值特征，30M 步）。目标是在数据积累到
> 足够规模之前，先用有限的真人数据 + 可再生的专家数据，稳定地逼近并超越它。

---

## 0. 核心判断（先说结论）

真人数据少时，不要指望"数据增强 + 大 BC"一条路走到底。这里有两个**本规则的硬约束**，决定了哪些增强能做：

| 增强 | 可行性 | 原因 |
| --- | --- | --- |
| 花色轮换（×3） | ❌ 不可行 | 小鸡是**具体的"1 条"（tile=18）**，不是字牌；换花色会改变小鸡身份，规则语义被破坏 |
| 座位轮换（×4） | ⚠️ 可行但要重建 | 四家手牌都在 DB 的 `privateViewJson` 里（AI 座位也记录），但导出轨迹目前只含人类座位手牌，需跨座位重建完整牌局 |
| 专家数据生成 | ✅ 立即可行 | heuristic/win_first agent 可无限生成"合理但偏弱"的示范，用作 BC 预训练 |

因此数据不足阶段的策略是**三条腿**：

```text
专家数据预训练（立即可做，无限量）
      ↓
真人数据 BC 微调（有限量，保"像人"）
      ↓
课程 PPO 提升强度（数据自产，不依赖真人）
      ↓
定时拉取累积真人数据 → 迭代重训
```

---

## 1. 数据保鲜（已落地，防数据过期浪费）

- 线上 DB 只保留已结束房间 7 天，过期即删。**已创建 Windows 定时任务**：
  - 任务名 `Mahjong-FlyBird-PullHumanData`，每天 04:00 以 SYSTEM 运行。
  - 执行 `scripts/run_pull.bat` → `scripts/pull_human_data.py --merge-into artifacts/human_traces.jsonl`。
  - 每次按 `(game_id, step)` 去重后增量合并，日志写入 `artifacts/pull.log`。
- 手动拉取/监控：
  ```bash
  python scripts/pull_human_data.py --stats          # 只看 DB 里真人步数
  python scripts/pull_human_data.py --merge-into artifacts/human_traces.jsonl   # 立即拉一次
  ```
- 密钥放在 `training/.secrets/backend.env`（已 gitignore），不外泄。

---

## 2. 阶段 A：专家数据预训练（立即可做，不依赖新真人数据）

用本地已有的 `mahjong_ai/imitation/collect_heuristic_dataset.py` 生成大规模启发式示范：

1. **收集**：heuristic（+win_first）专家对局，目标 50k–200k 条示范。
2. **BC 预训练**：让模型先学会"合法、不无脑杠、向听推进"的基线打法。
3. **真人 BC 微调**：用 386 条真人轨迹在预训练模型上做少量 epoch，注入"像人"的弃牌/副露偏好。
4. **PPO 课程**：`win_first → heuristic`（runbook §6.2），用现有 shaped reward。

注意：`collect_heuristic_dataset.py` 当前只存 `static` 观测。要对 **table-attention** 模型做 BC，需同步保存 `table` token（env 开了 `include_table` 后 obs 是 `{static, table}` 字典）。这是一个小改动，属于阶段 A 的第一步。

> 为什么这条最优先：真人 386 条只够"微调偏好"，不够"学会打麻将"。专家数据把"会打"这件事用近乎免费的数据解决，真人数据只负责"风格纠偏"。

## 3. 阶段 B：真人数据增强（座位轮换，中优先级）

- 花色轮换**不要做**（见上表，会破坏小鸡语义）。
- 座位轮换（×4）是麻将的合法对称（循环轮转保持出牌顺序）。但它要求**四家暗手**都在手：
  - 好消息：DB 的 `GameStep` 记录了所有座位（AI 236 步、HUMAN 73 步、FALLBACK 5 步），都带 `privateViewJson`。
  - 工作：写一个"按 `gameId` 跨座位重建完整牌局 → 轮转四家 → 重新生成 seat-0 观测"的脚本，把每条人类轨迹扩成 4 条。
- 收益：386 → ~1500 条等效轨迹，且不破坏规则语义。建议在真人数据 ≥500 条后再做，避免为了增强而过度工程。

## 4. BC 训练技巧（数据少时的正确用法）

- **类别再平衡**：DISCARD 占 84%（324/386），PONG/CHOW/PASS 极少。对稀有动作 oversample，或按 `1/类别频率` 加权，否则模型只学弃牌。
- **更多 epoch + 标签平滑**：数据少时 BC 易过拟合，用 `label_smoothing≈0.05`、早停于验证集。
- **留出验证集**：按 `gameId`（不是按条）切 10% 做验证，报 **top-1 / top-3 / top-5 动作命中率**。
- **学习率**：BC 用 `1e-4`；PPO 微调 `3e-6`（BC 初始化后），纯 PPO `1e-4`。
- **不要**：数据少时开 BC-aux（已在 240k 实验证实有害）、不要在续训时改奖励（value 失配）。

## 5. PPO 与评估（数据自产，不依赖真人）

- PPO 的 on-policy 数据由 env 自产，真人数据只影响 BC 初始化，所以**数据少不阻碍 PPO**。
- 保持 dense shaping（稀疏奖励 60k 步学不动，已证实）。当前合并后的奖励已含：危险牌、打牌效率、价值顺序、副露、杠判断、手牌目标。
- 评估统一：对 heuristic 打 **500–1000 局、≥3 seed**，指标 `win_rate / avg_score / deal_in / kong_rate / avg_steps`，噪声大时别下结论。

## 6. 分阶段实验路线图

| 阶段 | 内容 | 依赖 | 目标 |
| --- | --- | --- | --- |
| A1 | 扩展专家采集器保存 table token；生成 100k 专家示范 | 无 | 有可训练数据 |
| A2 | 专家 BC 预训练 → 真人 BC 微调（386）→ 课程 PPO | A1 | win_rate ≥ 0.17（追平现最好） |
| A3 | 系数消融 + 多 seed，锁定最优配置 | A2 | 稳定超过 heuristic |
| B1 | 跨座位重建 + 座位轮换增强脚本 | 真人 ≥500 条 | 数据 ×4 |
| B2 | 增强数据重训，对比 A 阶段 | B1 | 进一步提升 |
| C | 定时拉取累积 → 每 N 条重训 → 部署 A/B | 持续 | 超过线上 v3-lite 并替换 |

## 7. 运维与监控

- 每天看 `artifacts/pull.log` 与 `--stats` 输出，确认真人数据在增长、去重正确。
- 数据版本：`human_traces.jsonl` 是唯一规范数据集（已 git 跟踪），每积累一批可提交一个 `data(training)` 快照。
- 每轮训练产出 `metadata.json` + 评估指标，回填 `RESULTS.md`。

## 8. 一句话总结

**数据少时：用专家数据把"会打麻将"解决掉，用真人数据只解决"打得像人"，用课程 PPO 解决"打得赢"，用定时任务让真人数据持续流入、不再过期浪费。**
