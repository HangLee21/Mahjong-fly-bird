# Mahjong AI 训练文档（Training Runbook）

本文档用于在一台更好的机器（Linux / 多核 / 多卡）上继续训练“飞小鸡麻将”的
注意力模型，并与线上后端推理服务对接。

## 1. 目标定位

这是一个麻将小游戏 AI，要求：

- 更接近人类的打法：胡牌轮次短、不拖节奏。
- 有防守意识：不放炮、会读牌。
- 不无脑明杠。
- 对局胜率/总得分率高于 heuristic 基线。

## 2. 环境准备

已在一台 `RTX 4060 Laptop GPU` 上验证过的版本组合（版本必须匹配）：

```bash
python -m pip install \
  torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install --no-deps \
  stable-baselines3==2.3.2 sb3-contrib==2.3.0 gymnasium==0.29.1
```

注意：

- `sb3-contrib 2.9.x` 要求 `torch>=2.8`，会强制升级并破坏 CUDA torch，勿用。
- 更好机器上可换更新的 CUDA torch 与配套 sb3 版本，但需保证二者兼容。

其余依赖见 `training/requirements.txt`。

## 3. 代码结构

```text
training/
  mahjong_ai/
    env/
      actions.py          # 动作编码、动作掩码
      observation.py      # 静态观测 + 动作价值特征 + 牌面 token + 历史
      reward.py           # 危险牌惩罚、明杠惩罚、向听/胡牌 shaping
      gym_env.py          # 单智能体环境 + 对手
    rules/
      flybird.py          # 规则引擎 / GameState / 公共信息
      shanten.py          # 向听数、有效牌、牌型价值
    models/
      feature_extractor.py # TableAttentionTransformerExtractor / HybridHistoryTransformer / MLP
    agents/               # heuristic / win_first / random / model / pool
    train/train_ppo.py    # PPO 入口
    eval/evaluate.py      # 离线评估
    inference/predictor.py
  scripts/
    dump_human_steps.sql        # 从线上 DB 导出人类对局步骤
    export_human_traces.py      # 重建训练观测（static + table）
    train_bc_then_ppo.py        # BC 初始化 + PPO 微调（支持续训）
  configs/                      # 各类训练配置
```

## 4. 数据管线（人类打法注入）

线上人类固定坐 1 号位（seatIndex 0）。先导出，再转成训练轨迹：

```bash
psql "$DATABASE_URL" -t -A -f scripts/dump_human_steps.sql > human_steps.jsonl

python scripts/export_human_traces.py \
  --input human_steps.jsonl \
  --output artifacts/human_traces.jsonl
```

注意：

- 线上后端要保留已结束房间足够久，否则人类数据会被清理掉。生产环境建议设置
  `ROOM_FINISHED_TTL_MS >= 604800000`（7 天）。
- 当前导出轨迹的 `observation` 是 **394 维静态 + 4×76 牌面 token**；
  若开启动作价值特征后要做 BC，需要同步扩展导出脚本以重建动作特征。

## 5. 模型架构（关键结论）

新模型观测 = 静态观测 + 动作价值特征 + 牌面 token +（可选）动作历史。

### 5.1 静态观测

手牌、自家副露、四家弃牌/副露、分数、庄家/当前/相对位置、最后弃牌、剩余墙、
杠池、小鸡是否失效，共 394 维。

### 5.2 动作价值特征（最重要，不能丢）

线上 v3 模型 `obs_include_action_features: true`。它对每个合法动作模拟
`rule_adapter.step`，得到打后向听数、有效牌数、牌型价值、分数变化，编码为
`128×18` 的特征矩阵，展平后拼入静态观测（静态从 394 扩到 2698）。

这是 v3 比 heuristic 强的最主要来源：策略被直接告知“每个动作打完后的收益”，
不需要从原始牌面自行推断。

### 5.3 牌面 token + 注意力

每家编码为 76 维 token（弃牌、副露、分数、庄/当前、相对位置、手牌数），
`TableAttentionTransformerExtractor` 用 Transformer 对 4 个座位做注意力，
与静态 MLP、历史 Transformer 融合。

### 5.4 特征提取器

`training/mahjong_ai/models/feature_extractor.py`：

- `LayerNormMLPExtractor`：纯静态 MLP。
- `HybridHistoryTransformerExtractor`：静态 + 动作历史注意力。
- `TableAttentionTransformerExtractor`：静态 + 牌面注意力 +（可选）历史注意力。

## 6. 训练流程

推荐流程：**BC 初始化 → 课程 PPO → 对 heuristic 微调**。

### 6.1 BC 初始化

```bash
python scripts/train_bc_then_ppo.py \
  --config configs/ppo_table_attention_action.yaml \
  --bc-data artifacts/human_traces.jsonl \
  --bc-epochs 10 --bc-batch-size 256 \
  --output-dir artifacts/checkpoints/run_v4
```

### 6.2 PPO 微调（课程）

先打弱/中等对手拿到胡牌信号，再打 heuristic：

```bash
# 1) win_first 课程
python scripts/train_bc_then_ppo.py \
  --config configs/ppo_table_attention_winfirst.yaml \
  --bc-data artifacts/human_traces.jsonl \
  --resume artifacts/checkpoints/run_v4/bc_model.zip \
  --output-dir artifacts/checkpoints/run_v4_winfirst

# 2) heuristic 微调
python scripts/train_bc_then_ppo.py \
  --config configs/ppo_table_attention_action.yaml \
  --bc-data artifacts/human_traces.jsonl \
  --resume artifacts/checkpoints/run_v4_winfirst/final_model.zip \
  --output-dir artifacts/checkpoints/run_v4_final
```

### 6.3 纯 PPO（无 BC）

```bash
python scripts/train_bc_then_ppo.py \
  --config configs/ppo_table_attention_action.yaml \
  --no-bc \
  --output-dir artifacts/checkpoints/run_v4_ppo_only
```

## 7. 配置说明

| 配置 | 用途 |
| --- | --- |
| `ppo_table_attention_action.yaml` | **推荐**：动作价值特征 + 牌面注意力 |
| `ppo_table_attention.yaml` | 200k 步档（注意加动作特征） |
| `ppo_table_attention_winfirst.yaml` | 中等难度课程 |
| `ppo_table_attention_pool.yaml` | 混合对手池实验 |
| `ppo_table_attention_medium/minimal/winboost/long.yaml` | 各类消融实验 |
| `ppo_table_attention_smoke.yaml` | 冒烟测试 |

## 8. 评估

```bash
python -m mahjong_ai.eval.evaluate \
  --model artifacts/checkpoints/run_v4_final/final_model.zip \
  --config configs/ppo_table_attention_action.yaml \
  --num-games 300 --opponent heuristic
```

关键指标：

- `win_rate`：对 heuristic 胜率。
- `avg_score`：平均每局得分。
- `avg_steps`：平均步数（近似胡牌轮次）。
- `deal_in_rate`：放炮率。
- `kong_rate`：明杠率。

heuristic 基线（4 家对打，seat 0 视角）：

- `win_rate ≈ 0.165`、`avg_steps ≈ 13.22`、`deal_in ≈ 0.12`、`kong_rate ≈ 0.026`。

当前最优 checkpoint 对 heuristic 约 17%（噪声大），`kong_rate=0`、`avg_steps` 略快。

## 9. 换更好机器的并行训练

动作价值特征会让 env 步进慢约 10 倍，强烈建议并行：

```yaml
train:
  vec_env_type: subproc
  num_envs: 8          # 按 CPU 核数调整
  n_steps: 4096
  batch_size: 8192
  learning_rate: 0.000003
```

Linux 下 `subproc` 更稳定；Windows 下 `subproc` 可能卡死，需要排查
`multiprocessing` spawn + env factory 的 pickling。

大机器建议：

- `num_envs 8-16`。
- 动作价值特征开启（`include_action_features: true`）。
- lr 从 `3e-6` 开始（微调 BC 模型），纯 PPO 可用 `1e-4`。
- 训练量目标至少数百万步（线上 v3 是 3000 万步）。

## 10. 已知坑

- **动作价值特征是 v3 强的关键，新模型若关闭它，胜率会明显下降。**
- 动作价值特征显著拖慢 env，需要并行环境。
- BC 数据太少（如只有几百条）会让策略偏弱；需要持续积累人类数据。
- 续训时**不要改奖励**，否则 value 网络失配会退化；奖励迭代要全新训练。
- 明杠惩罚很有效（`kong_rate` 可到 0）。
- 危险牌惩罚有助于降低放炮率。

## 11. 部署对接

训练完成后：

1. 把 `checkpoint` 拷贝到后端 `model/` 目录。
2. 把训练仓库 `mahjong_ai/env/observation.py` 与 `models/feature_extractor.py`
   同步到后端 `mahjong_ai/`。
3. 更新后端 `ai_service/server.py` 的模型路径与观测配置。
4. 重启 AI 服务，用 `evaluate` 或线上 A/B 对比验证。

详见 `RESULTS.md` 中的实验结论与指标矩阵。
