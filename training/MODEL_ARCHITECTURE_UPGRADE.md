# 模型架构升级设计（Attention + 防守 + 大牌意识）

> 目标：把当前"MLP + 计数向量"的简单架构，升级为能捕捉牌型结构、对手动态、
> 防守风险与大牌价值的多头注意力架构。本文档给出可落地的模块级设计，
> 并把每个目标映射到具体的网络结构 / 监督信号 / 奖励调整。

---

## 0. 现状与根本不足

线上 v3-lite 部署的是 `LayerNormMLPExtractor`：一个 MLP 吃 2698 维向量
（394 静态 + 128×18 动作价值特征），输出 policy/value。问题：

1. **手牌是 34 维计数向量**，丢失了结构——模型分不清"7/8/9 顺子"和"跨花色散牌 7、8、9"，
   更分不清"对子/搭子/孤张"。
2. **没有对手建模**：只知道对手弃了哪些牌，不知道"谁在听牌、打哪张危险"。
3. **没有显式风险/价值输出**：防守和大牌全靠 reward shaping 间接学，慢且不稳。
4. **吃碰无门控**：副露决策只看"是否推进向听"，不看"推进的是不是一手大牌"。

好消息：上游已新增 `TableAttentionTransformerExtractor`（4 座位 token 注意力），
且数据库 `resultJson` 里有**放炮者 / 胡牌番型 / 得分**，人类轨迹里有**吃碰/过牌决策**，
这些标签足够训练下面的辅助头。

---

## 1. 目标 → 机制映射

| 你的诉求 | 架构机制 | 监督信号 |
| --- | --- | --- |
| 用 attention 捕捉更多内容 | 手牌逐张 token 自注意力 + 手牌↔牌河交叉注意力 | 自监督（结构） |
| 防守策略 | Risk Head（每张牌放炮概率）+ 危险牌特征 | `resultJson.loserIndexes` + 胡牌 tile |
| 避免无脑吃碰 | Claim Gate（吃碰/过牌二分类门控） | 人类轨迹的 PONG/CHOW/PASS 决策 |
| 做大牌意识 | HandValue Head（预测最终番/分）+ 番型倾向特征 | `resultJson.winnerDetails.points/fanItems` |

---

## 2. 升级后的整体架构

```text
                        ┌──────────────────────────────────────────────┐
                        │                 Observation                 │
                        │  static(394+goal/danger)  action_features   │
                        │  hand_tokens(14×d)  table_tokens(4×d)  hist │
                        └──────┬──────────┬──────────┬─────────┬──────┘
                               │          │          │         │
                    ┌──────────▼──┐  ┌────▼─────┐  ┌─▼─────────▼──┐
                    │ StaticEncoder│  │ HandEnc  │  │ TableEncoder │
                    │   (MLP)      │  │(Tile-Att)│  │ (Seat-Att)   │
                    └──────────┬──┘  └────┬─────┘  └──────┬───────┘
                               │         │  hand_struct    │ table_feat
                               │         └───────┬────────┘
                               │          ┌──────▼──────────────┐
                               │          │ CrossAttention(hand │
                               │          │  queries→discards)  │
                               │          │   => danger_feat    │
                               │          └──────┬──────────────┘
                               └─────────┬───────┴───────┬────────┘
                                     ┌───▼────────────────▼───┐
                                     │   Feature Fusion (MLP) │
                                     │        latent z        │
                                     └──┬───┬───┬───┬───┬────┘
                ┌───────────┬───────────┘   │   │   │   │
        ┌───────▼──────┐ ┌──▼────────┐ ┌─────▼─┐ ┌▼──────────┐ ┌▼─────────┐
        │ Policy Head  │ │ Value Head│ │RiskHd │ │HandValueHd│ │ClaimHead │
        │ (legal action)│ │ V(s)      │ │34 risk│ │ fan/pts   │ │P(claim)  │
        └──────────────┘ └───────────┘ └───────┘ └───────────┘ └──────────┘
```

- **Policy/Value**：PPO 主目标（MaskablePPO，动作掩码保证合法）。
- **Risk/HandValue/Claim**：辅助头，训练时加辅助损失，推理时用于**决策门控与奖励增强**，
  不改变动作合法性。

---

## 3. 关键新模块设计

### 3.1 Hand Encoder（逐张牌注意力）——"捕捉更多内容"的核心

把 14 张手牌编码成序列 token，跑一个小 Transformer：

- 每张牌 token = `tile_embedding(34→d)` + `suit(4)` + `rank(9)` + `is_honor` + `is_xiaoji`。
- 可加"手牌计数"作为可学习的位置/重复编码（同牌多张）。
- `TransformerEncoder(d, nhead, 2-3 层)` → 取 mean-pool 得 `hand_struct`。

**为什么比计数向量强**：注意力天然表达"这张 8 万和 6、7 万是否相邻成搭子"，
"这副牌是否缺一门可冲清一色"。这是"做大牌意识"的感知基础。

### 3.2 Table Encoder（座位注意力，增强版）——对手建模

在现有 `TableAttentionTransformerExtractor` 基础上，把每个座位 token 从 76 维
**扩充**加入防守特征：

- 该家每门花色已弃张数（3 维）、已见安全牌数、`handCount`、副露数/类型。
- 危险度代理：`discard_danger_score`（已有函数）对每张牌的估值。
- 一个额外的 `[TENPAI]` 可学习标记，让注意力显式聚合"谁最可能听牌"。

### 3.3 Cross-Attention Danger（手牌 ↔ 牌河）——防守

- Query = 手牌 token（`hand_struct`），Key/Value = 三家弃牌+副露 token。
- 输出每张手牌的"被需要度" `danger_feat`。
- 这个特征直接决定"哪张牌打出去危险"，喂给 Risk Head 和 Policy Head。

### 3.4 Risk Head（防守头）——显式放炮概率

- 输出 34 维 `P(deal_in | discard tile)`。
- 监督：`resultJson` 里 `reason=ron` 时，`loserIndexes` 是放炮者，`winnerDetails.tile`
  是胡的那张——可回标"这一打放炮"；其余打牌为负样本。
- 损失：`BCE(risk, deal_in_label)`，对放炮样本加大权重（正样本稀少）。
- 用法：推理时用 `risk` 做**软惩罚**（risk 高的牌 logits 下调），并在落后/对手疑似
  听牌时自动抬高防守权重。

### 3.5 HandValue Head（大牌头）——显式番/分预测

- 输出：预测这手牌最终能胡的番数档位（如 0/1/2/3+ 或回归 points）。
- 监督：`resultJson.winnerDetails.points`（赢家），非赢家（平胡输/流局）标 0 或
  用 `score_hand` 的**当前最佳价值**做 soft label。
- 用法：
  1. 作为观察特征回灌（"我这手牌值多少"），让 policy 决策时"做大牌"。
  2. 奖励增强：`fan_bonus` 改为**与预测价值成比例**，大牌胡牌奖励显著高于平胡。

### 3.6 Claim Head（吃碰门控）——避免无脑吃碰

- 输出：当前局面下"吃/碰/过"的概率。
- 监督：人类轨迹里 `PONG / CHOW_LEFT / CHOW_MIDDLE / CHOW_RIGHT / PASS` 决策
  （当前 386 条里有 26 次吃碰、32 次过牌，后续数据持续增加）。
- 用法：对 `PONG/CHOW` 动作施加门控——若 `P(claim)` 低或"吃了也只是平胡"
  （结合 HandValue Head），则压低吃碰 logits。目标是"该吃才吃，不该吃就过"。

---

## 4. 奖励调整（配合架构）

| 现状 | 问题 | 调整 |
| --- | --- | --- |
| `terminal_win_bonus` 恒定 | 平胡和大牌奖励一样 → 无脑平胡 | 改成 `win_bonus × (1 + k·fan)`，大牌权重更高 |
| `fan_bonus/point_bonus` 已存在但偏小 | 大牌意识弱 | 提高比例，或直接用 HandValue Head 输出 |
| 吃碰奖励只看向听 | 不看手牌价值 | 吃碰后若 HandValue 下降则惩罚 |
| 危险牌惩罚 `discard_danger_penalty` | 只有 reward，无显式信号 | Risk Head 提供结构化监督，reward 与之对齐 |

---

## 5. 数据与标签（已确认可用）

| 标签 | 来源 | 当前量 |
| --- | --- | --- |
| 人类吃碰/过牌决策 | `human_traces.jsonl` 的 action 类型 | 58 次（含 PASS） |
| 放炮者 + 胡牌 tile | `Game.resultJson`（`reason/loserIndexes/winnerDetails.tile`） | 随对局增长 |
| 胡牌番型/得分 | `Game.resultJson.winnerDetails.fanItems/points` | 随对局增长 |
| 手牌结构（自监督） | 训练环境实时生成，无需标签 | 无限 |

> 导出脚本需扩展：除了重建静态观测，还要重建上面的**防守/大牌标签**（从 `resultJson`
> 回标到每个 `GameStep`）。这是架构升级的前置数据工作。

---

## 6. 分阶段实施（按性价比/风险排序）

> **进度**：✅ P0 已完成；✅ P1 已部分完成（番数比例奖励 + 牌型目标特征）。
> HandValue 辅助头（依赖 `resultJson` 番型标签）待数据增长后接。

| 阶段 | 内容 | 改动文件 | 状态 |
| --- | --- | --- | --- |
| **P0 感知增强** | Hand Encoder（逐张牌注意力）+ 观测加入 `hand_goal_scores`（4 维）+ 危险牌特征 | `observation.py` `feature_extractor.py` | ✅ 已实现（`mahjong_attention` 提取器） |
| **P1 大牌意识** | 番数比例化奖励 `win_fan_scale` + 牌型目标特征进观测 | `reward.py` `observation.py` | ✅ 已实现；HandValue 辅助头待数据 |
| **P2 防守** | Risk Head + 交叉注意力 + 放炮标签回标 | `heads.py`(新) `reward.py` `train_ppo.py` | ⏳ 待做 |
| **P3 吃碰门控** | Claim Head + 吃碰 logits 门控 | 同上 | ⏳ 待做 |
| **P4 融合打磨** | 多任务损失权重、消融、评估（防守/大牌指标） | 各文件 | ⏳ 待做 |

### 已落地实现（P0 + P1）

- `observation.py`：新增 `build_hand_tokens`（14×38 逐张牌 token + mask）、
  `include_hand_observation`、`obs_include_hand_goal`（把 standard/七对/清一色/大对
  4 个牌型目标分数拼进 static，394→398 维）。
- `feature_extractor.py`：新增 `MahjongAttentionExtractor`——手牌 Transformer
  （padding mask）+ 座位 Transformer + 静态 MLP，三路融合。
- `reward.py`：`_terminal_win_shaping` 支持 `win_fan_scale`，胡牌奖励随番数
  比例放大（默认 0，不改变旧配置）。
- `gym_env.py` / `train_ppo.py`：Dict 观测空间接入 `hand`/`hand_mask`，提取器路由
  `mahjong_attention`。
- 配置：`configs/ppo_mahjong_attention_smoke.yaml`；测试：`tests/test_mahjong_attention.py`。

> 每阶段独立可验证：P0 看"是否更快学会向听推进"，P1 看"胡牌番均值是否上升"，
> P2 看 `deal_in_rate` 下降，P3 看"吃碰率是否更接近人类"。

---

## 7. 新增评估指标（对齐你的诉求）

- 防守：`deal_in_rate`、`avg_discard_danger`（打出的牌的平均危险度）、落后时防守强度。
- 大牌：`avg_win_fan`（胡牌平均番数）、`big_hand_rate`（番 ≥2 的占比）。
- 吃碰：`claim_rate` 与人类分布对比、`claim_value`（吃碰后手牌价值变化）。

---

## 8. 与现有代码的接缝

- `feature_extractor.py`：新增 `MahjongAttentionExtractor`（含 Hand/Table/Cross 编码器 + 多任务头），
  与现有 4 个提取器并存，通过 `config.model.feature_extractor.name` 切换。
- `observation.py`：`build_static_observation` 追加 goal/danger 特征；`build_hand_tokens`（新）。
- `train_ppo.py` / `train_bc_then_ppo.py`：多任务辅助损失接入（BC 数据里的标签一起消费）。
- `export_human_traces.py`：回标放炮/番型标签，供辅助头监督。
- `reward.py`：fan 比例化 + 吃碰价值惩罚。

---

## 9. 结论

可以，而且信号是齐的。最优先、最低风险的一步是 **P0：把逐张牌注意力编码器做出来，
并把牌型目标分数和危险牌特征显式喂进观测**——这一步立刻让模型"看得见结构"，
后面 P1/P2/P3 的防守与大牌头都是在这条骨架上的增量。
