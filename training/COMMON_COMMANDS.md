# 常用训练与评估指令

以下命令默认在 `D:\Mahjong-flybird\training` 目录下执行。

```powershell
cd D:\Mahjong-flybird\training
```

## V2 1500w 基线模型

模型路径：

```text
artifacts/checkpoints/ppo_v2_committed_scratch/periodic/model_15000000_steps.zip
```

### 评估 1500w vs heuristic

```powershell
python -m mahjong_ai.eval.evaluate `
  --model artifacts/checkpoints/ppo_v2_committed_scratch/periodic/model_15000000_steps.zip `
  --num-games 5000 `
  --opponent heuristic `
  --output artifacts/reports/eval_v2_1500w_heuristic.json
```

### 评估 1500w vs pool

```powershell
python -m mahjong_ai.eval.evaluate `
  --model artifacts/checkpoints/ppo_v2_committed_scratch/periodic/model_15000000_steps.zip `
  --num-games 5000 `
  --opponent pool `
  --opponent-pool-config configs/ppo_v2_from_scratch_committed.yaml `
  --output artifacts/reports/eval_v2_1500w_pool.json
```

### 导出 1500w vs heuristic 单局牌谱

```powershell
python -m mahjong_ai.eval.play_game `
  --model artifacts/checkpoints/ppo_v2_committed_scratch/periodic/model_15000000_steps.zip `
  --config configs/ppo_v2_from_scratch_committed.yaml `
  --seed 20260523 `
  --opponent heuristic `
  --output artifacts/reports/game_v2_1500w_vs_heuristic.txt `
  --json-output artifacts/reports/game_v2_1500w_vs_heuristic.json
```

### 导出 1500w vs pool 单局牌谱

```powershell
python -m mahjong_ai.eval.play_game `
  --model artifacts/checkpoints/ppo_v2_committed_scratch/periodic/model_15000000_steps.zip `
  --config configs/ppo_v2_from_scratch_committed.yaml `
  --seed 20260523 `
  --opponent pool `
  --opponent-pool-config configs/ppo_v2_from_scratch_committed.yaml `
  --output artifacts/reports/game_v2_1500w_vs_pool.txt `
  --json-output artifacts/reports/game_v2_1500w_vs_pool.json
```

## V2 Shape V2 训练

这是当前用于修正拆搭子、弱杠牌、并针对 heuristic 的 finetune 配置。

配置文件：

```text
configs/ppo_v2_beat_heuristic_shape_v2.yaml
```

### 从 V2 1500w 开始训练

```powershell
python -m mahjong_ai.train.train_ppo `
  --config configs/ppo_v2_beat_heuristic_shape_v2.yaml `
  --resume artifacts/checkpoints/ppo_v2_committed_scratch/periodic/model_15000000_steps.zip
```

### 评估 Shape V2 某个 checkpoint vs heuristic

把 `XXXX` 替换成实际步数，例如 `16000000`、`17000000`。

```powershell
python -m mahjong_ai.eval.evaluate `
  --model artifacts/checkpoints/ppo_v2_beat_heuristic_shape_v2/periodic/model_XXXX_steps.zip `
  --num-games 5000 `
  --opponent heuristic `
  --output artifacts/reports/eval_v2_shape_v2_XXXX_heuristic.json
```

### 评估 Shape V2 某个 checkpoint vs pool

```powershell
python -m mahjong_ai.eval.evaluate `
  --model artifacts/checkpoints/ppo_v2_beat_heuristic_shape_v2/periodic/model_XXXX_steps.zip `
  --num-games 5000 `
  --opponent pool `
  --opponent-pool-config configs/ppo_v2_beat_heuristic_shape_v2.yaml `
  --output artifacts/reports/eval_v2_shape_v2_XXXX_pool.json
```

### 导出 Shape V2 单局牌谱

```powershell
python -m mahjong_ai.eval.play_game `
  --model artifacts/checkpoints/ppo_v2_beat_heuristic_shape_v2/periodic/model_XXXX_steps.zip `
  --config configs/ppo_v2_beat_heuristic_shape_v2.yaml `
  --seed 20260523 `
  --opponent heuristic `
  --output artifacts/reports/game_v2_shape_v2_XXXX_vs_heuristic.txt `
  --json-output artifacts/reports/game_v2_shape_v2_XXXX_vs_heuristic.json
```

## 历史候选模型

## V2 Honor Fix 训练

这是当前用于修正“过度保留孤张字牌、打掉数牌搭子/对子”的 finetune 配置。

配置文件：

```text
configs/ppo_v2_beat_heuristic_honor_fix.yaml
```

### 从 V2 1500w 开始训练 Honor Fix

```powershell
python -m mahjong_ai.train.train_ppo `
  --config configs/ppo_v2_beat_heuristic_honor_fix.yaml `
  --resume artifacts/checkpoints/ppo_v2_committed_scratch/periodic/model_15000000_steps.zip
```

### 评估 Honor Fix 某个 checkpoint vs heuristic

把 `XXXX` 替换成实际步数，例如 `16000000`、`17000000`。

```powershell
python -m mahjong_ai.eval.evaluate `
  --model artifacts/checkpoints/ppo_v2_beat_heuristic_honor_fix/periodic/model_XXXX_steps.zip `
  --num-games 5000 `
  --opponent heuristic `
  --output artifacts/reports/eval_v2_honor_fix_XXXX_heuristic.json
```

### 导出 Honor Fix 单局牌谱

```powershell
python -m mahjong_ai.eval.play_game `
  --model artifacts/checkpoints/ppo_v2_beat_heuristic_honor_fix/periodic/model_XXXX_steps.zip `
  --config configs/ppo_v2_beat_heuristic_honor_fix.yaml `
  --seed 20260523 `
  --opponent heuristic `
  --output artifacts/reports/game_v2_honor_fix_XXXX_vs_heuristic.txt `
  --json-output artifacts/reports/game_v2_honor_fix_XXXX_vs_heuristic.json
```

## V2.5 行为克隆 + PPO 微调

V2.5 的目标是先用 heuristic 专家样本训练基础牌效，再用 PPO 微调超过 heuristic。

配置文件：

```text
configs/bc_v25_heuristic.yaml
configs/ppo_v25_bc_finetune.yaml
```

### 1. 采集 heuristic 专家样本

默认采集 `100w` 条样本到：

```text
artifacts/datasets/v25_heuristic_bc_100w.npz
```

```powershell
python -m mahjong_ai.imitation.collect_heuristic_dataset `
  --config configs/bc_v25_heuristic.yaml
```

采集时会每 `10000` 条样本输出一次进度，包括样本数、局数、速度、已用时间和 ETA。也可以临时指定间隔：

```powershell
python -m mahjong_ai.imitation.collect_heuristic_dataset `
  --config configs/bc_v25_heuristic.yaml `
  --progress-interval 5000
```

### 2. 行为克隆训练 BC 模型

```powershell
python -m mahjong_ai.imitation.train_behavior_clone `
  --config configs/bc_v25_heuristic.yaml
```

输出模型：

```text
artifacts/checkpoints/v25_bc_heuristic/bc_model.zip
```

### 3. 从 BC 模型开始 PPO 微调

```powershell
python -m mahjong_ai.train.train_ppo `
  --config configs/ppo_v25_bc_finetune.yaml `
  --resume artifacts/checkpoints/v25_bc_heuristic/bc_model.zip
```

### 4. 评估 V2.5 某个 checkpoint

```powershell
python -m mahjong_ai.eval.evaluate `
  --model artifacts/checkpoints/ppo_v25_bc_finetune/periodic/model_XXXX_steps.zip `
  --num-games 5000 `
  --opponent heuristic `
  --output artifacts/reports/eval_v25_bc_finetune_XXXX_heuristic.json
```

### 5. 导出 V2.5 牌谱

```powershell
python -m mahjong_ai.eval.play_game `
  --model artifacts/checkpoints/ppo_v25_bc_finetune/periodic/model_XXXX_steps.zip `
  --config configs/ppo_v25_bc_finetune.yaml `
  --seed 20260523 `
  --opponent heuristic `
  --output artifacts/reports/game_v25_bc_finetune_XXXX_vs_heuristic.txt `
  --json-output artifacts/reports/game_v25_bc_finetune_XXXX_vs_heuristic.json
```

### V1 1000w vs heuristic

```powershell
python -m mahjong_ai.eval.evaluate `
  --model artifacts/checkpoints/ppo_v1_committed_scratch/periodic/model_10000000_steps.zip `
  --num-games 5000 `
  --opponent heuristic `
  --output artifacts/reports/eval_v1_1000w_heuristic.json
```

### V2 1900w pool strong vs heuristic

```powershell
python -m mahjong_ai.eval.evaluate `
  --model artifacts/checkpoints/ppo_v2_human_finetune_high_model_gpu_opponents_stable/periodic/model_19000000_steps.zip `
  --num-games 5000 `
  --opponent heuristic `
  --output artifacts/reports/eval_v2_1900w_pool_strong_heuristic.json
```

### V2 1900w pool strong vs pool

```powershell
python -m mahjong_ai.eval.evaluate `
  --model artifacts/checkpoints/ppo_v2_human_finetune_high_model_gpu_opponents_stable/periodic/model_19000000_steps.zip `
  --num-games 5000 `
  --opponent pool `
  --opponent-pool-config configs/ppo_v2_human_finetune_high_model_gpu_opponents_stable.yaml `
  --output artifacts/reports/eval_v2_1900w_pool_strong_pool.json
```

## 查看 GPU 与训练进程

```powershell
nvidia-smi
```

```powershell
Get-CimInstance Win32_Process -Filter "name='python.exe'" |
  Select-Object ProcessId,CommandLine |
  Format-List
```

## 结果判断参考

优先看：

```text
avg_score       越高越好，目标先转正
win_rate        目标 0.25+
deal_in_rate    目标低于 0.10
draw_rate       不宜过高
xiaoji_discard_rate 越低越好，但不能为了不打小鸡乱拆牌
illegal/fallback/truncated 必须为 0
```

训练日志参考：

```text
approx_kl       0.005 - 0.025 比较健康
clip_fraction   0.05 - 0.30 可接受，长期高于 0.35 偏激进
ep_rew_mean     只能辅助参考，最终以 evaluate 为准
```

## V2.6 / V3 overnight commands

### V2.6 from scratch, no old model opponents

```powershell
python -m mahjong_ai.train.train_ppo `
  --config configs/ppo_v26_scratch_foundation.yaml
```

### V2.7 from scratch, taatsu value curriculum

```powershell
D:\MiniConda\python.exe -m mahjong_ai.train.train_ppo `
  --config configs/ppo_v27_scratch_taatsu_value.yaml
```

### Evaluate V2.7 scratch checkpoint

```powershell
D:\MiniConda\python.exe -m mahjong_ai.eval.evaluate `
  --model artifacts/checkpoints/ppo_v27_scratch_taatsu_value/periodic/model_XXXX_steps.zip `
  --num-games 5000 `
  --opponent heuristic `
  --output artifacts/reports/eval_v27_taatsu_XXXX_heuristic.json
```

### Play one readable V2.7 scratch game

```powershell
D:\MiniConda\python.exe -m mahjong_ai.eval.play_game `
  --model artifacts/checkpoints/ppo_v27_scratch_taatsu_value/periodic/model_XXXX_steps.zip `
  --config configs/ppo_v27_scratch_taatsu_value.yaml `
  --seed 20260523 `
  --opponent heuristic `
  --output artifacts/reports/game_v27_taatsu_XXXX_vs_heuristic.txt `
  --json-output artifacts/reports/game_v27_taatsu_XXXX_vs_heuristic.json
```

### V2.8 from scratch, claim/pass and ready protection

```powershell
D:\MiniConda\python.exe -m mahjong_ai.train.train_ppo `
  --config configs/ppo_v28_scratch_claim_ready.yaml
```

### Evaluate V2.8 scratch checkpoint

```powershell
D:\MiniConda\python.exe -m mahjong_ai.eval.evaluate `
  --model artifacts/checkpoints/ppo_v28_scratch_claim_ready/periodic/model_XXXX_steps.zip `
  --num-games 5000 `
  --opponent heuristic `
  --output artifacts/reports/eval_v28_claim_ready_XXXX_heuristic.json
```

### Play one readable V2.8 scratch game

```powershell
D:\MiniConda\python.exe -m mahjong_ai.eval.play_game `
  --model artifacts/checkpoints/ppo_v28_scratch_claim_ready/periodic/model_XXXX_steps.zip `
  --config configs/ppo_v28_scratch_claim_ready.yaml `
  --seed 20260523 `
  --opponent heuristic `
  --output artifacts/reports/game_v28_claim_ready_XXXX_vs_heuristic.txt `
  --json-output artifacts/reports/game_v28_claim_ready_XXXX_vs_heuristic.json
```

### Evaluate fixed scenario decisions

```powershell
D:\MiniConda\python.exe -m mahjong_ai.eval.evaluate_scenarios `
  --model artifacts/checkpoints/ppo_v28_scratch_claim_ready/evals/eval_model_10000000_steps.zip `
  --config configs/ppo_v28_scratch_claim_ready.yaml `
  --output artifacts/reports/scenarios_v28_1000w.txt `
  --json-output artifacts/reports/scenarios_v28_1000w.json
```

## V2.9 claim quality from scratch

V2.9 的目标是修正 V2.8 场景测试里暴露的副露问题：能过牌、减少无改善吃碰、强化暗杠/加杠收益，并继续保护听牌和一向听。

配置文件：

```text
configs/ppo_v29_scratch_claim_quality.yaml
```

### Train V2.9 from scratch

```powershell
D:\MiniConda\python.exe -m mahjong_ai.train.train_ppo `
  --config configs/ppo_v29_scratch_claim_quality.yaml
```

### Evaluate V2.9 checkpoint vs heuristic

把 `XXXX` 替换成实际步数，例如 `3000000`、`8000000`、`10000000`。

```powershell
D:\MiniConda\python.exe -m mahjong_ai.eval.evaluate `
  --model artifacts/checkpoints/ppo_v29_scratch_claim_quality/periodic/model_XXXX_steps.zip `
  --num-games 5000 `
  --opponent heuristic `
  --output artifacts/reports/eval_v29_claim_quality_XXXX_heuristic.json
```

### Evaluate V2.9 checkpoint vs pool

```powershell
D:\MiniConda\python.exe -m mahjong_ai.eval.evaluate `
  --model artifacts/checkpoints/ppo_v29_scratch_claim_quality/periodic/model_XXXX_steps.zip `
  --num-games 5000 `
  --opponent pool `
  --opponent-pool-config configs/ppo_v29_scratch_claim_quality.yaml `
  --output artifacts/reports/eval_v29_claim_quality_XXXX_pool.json
```

### Play one readable V2.9 game

```powershell
D:\MiniConda\python.exe -m mahjong_ai.eval.play_game `
  --model artifacts/checkpoints/ppo_v29_scratch_claim_quality/periodic/model_XXXX_steps.zip `
  --config configs/ppo_v29_scratch_claim_quality.yaml `
  --seed 20260523 `
  --opponent heuristic `
  --output artifacts/reports/game_v29_claim_quality_XXXX_vs_heuristic.txt `
  --json-output artifacts/reports/game_v29_claim_quality_XXXX_vs_heuristic.json
```

### Evaluate fixed scenario decisions for V2.9

```powershell
D:\MiniConda\python.exe -m mahjong_ai.eval.evaluate_scenarios `
  --model artifacts/checkpoints/ppo_v29_scratch_claim_quality/periodic/model_XXXX_steps.zip `
  --config configs/ppo_v29_scratch_claim_quality.yaml `
  --output artifacts/reports/scenarios_v29_claim_quality_XXXX.txt `
  --json-output artifacts/reports/scenarios_v29_claim_quality_XXXX.json
```

### V2.9 quick checkpoints to inspect

建议优先在 `3000000`、`8000000`、`10000000` 三个点检查：

```text
claim_pass_rate 是否明显大于 0
accepted_claim_regress_rate 是否下降
discard_regression_rate 是否低于 V2.8
scenarios 是否从 12/18 提升到 14/18+
avg_score 是否比 V2.8 的 -0.3526 更接近 0
```

### Evaluate V2.6 scratch checkpoint

```powershell
python -m mahjong_ai.eval.evaluate `
  --model artifacts/checkpoints/ppo_v26_scratch_foundation/periodic/model_XXXX_steps.zip `
  --num-games 5000 `
  --opponent heuristic `
  --output artifacts/reports/eval_v26_scratch_XXXX_heuristic.json
```

### Play one readable V2.6 scratch game

```powershell
python -m mahjong_ai.eval.play_game `
  --model artifacts/checkpoints/ppo_v26_scratch_foundation/periodic/model_XXXX_steps.zip `
  --config configs/ppo_v26_scratch_foundation.yaml `
  --seed 20260523 `
  --opponent heuristic `
  --output artifacts/reports/game_v26_scratch_XXXX_vs_heuristic.txt `
  --json-output artifacts/reports/game_v26_scratch_XXXX_vs_heuristic.json
```

## V3-lite action value model

V3-lite 不再只靠 reward 调参，而是在 observation 里加入每个合法动作执行后的牌效特征。当前实现会为 128 个 action 都提供一行动作特征，合法动作包含：

```text
动作类型、执行后向听、向听变化、形状分、分数变化、是否打小鸡/字牌/幺九、是否退向听
```

推荐路线是先 BC warmup，再 PPO finetune。当前默认关闭完整有效进张枚举，避免 CPU 被牌效搜索拖死；等确认 V3-lite 有收益后，再开启 `action_features.effective_tiles: true` 跑重版本。

配置文件：

```text
configs/bc_v3_lite_action_value.yaml
configs/ppo_v3_lite_action_value_bc_finetune.yaml
configs/ppo_v3_lite_action_value_scratch.yaml
```

### 1. Collect V3-lite heuristic dataset

```powershell
D:\MiniConda\python.exe -m mahjong_ai.imitation.collect_heuristic_dataset `
  --config configs/bc_v3_lite_action_value.yaml
```

### 2. Train V3-lite BC warmup model

```powershell
D:\MiniConda\python.exe -m mahjong_ai.imitation.train_behavior_clone `
  --config configs/bc_v3_lite_action_value.yaml
```

输出模型：

```text
artifacts/checkpoints/v3_lite_action_value_bc/bc_model.zip
```

### 3. PPO finetune from V3-lite BC model

```powershell
D:\MiniConda\python.exe -m mahjong_ai.train.train_ppo `
  --config configs/ppo_v3_lite_action_value_bc_finetune.yaml `
  --resume artifacts/checkpoints/v3_lite_action_value_bc/bc_model.zip `
  --reset-timesteps
```

### Alternative: Train V3-lite PPO from scratch

```powershell
D:\MiniConda\python.exe -m mahjong_ai.train.train_ppo `
  --config configs/ppo_v3_lite_action_value_scratch.yaml
```

### Evaluate V3-lite BC finetune checkpoint

```powershell
D:\MiniConda\python.exe -m mahjong_ai.eval.evaluate `
  --model artifacts/checkpoints/ppo_v3_lite_action_value_bc_finetune/periodic/model_XXXX_steps.zip `
  --config configs/ppo_v3_lite_action_value_bc_finetune.yaml `
  --num-games 5000 `
  --opponent heuristic `
  --output artifacts/reports/eval_v3_lite_bc_finetune_XXXX_heuristic.json
```

### Evaluate V3-lite scratch checkpoint

```powershell
D:\MiniConda\python.exe -m mahjong_ai.eval.evaluate `
  --model artifacts/checkpoints/ppo_v3_lite_action_value_scratch/periodic/model_XXXX_steps.zip `
  --config configs/ppo_v3_lite_action_value_scratch.yaml `
  --num-games 5000 `
  --opponent heuristic `
  --output artifacts/reports/eval_v3_lite_scratch_XXXX_heuristic.json
```

### Play one readable V3-lite BC finetune game

```powershell
D:\MiniConda\python.exe -m mahjong_ai.eval.play_game `
  --model artifacts/checkpoints/ppo_v3_lite_action_value_bc_finetune/periodic/model_XXXX_steps.zip `
  --config configs/ppo_v3_lite_action_value_bc_finetune.yaml `
  --seed 20260523 `
  --opponent heuristic `
  --output artifacts/reports/game_v3_lite_bc_finetune_XXXX_vs_heuristic.txt `
  --json-output artifacts/reports/game_v3_lite_bc_finetune_XXXX_vs_heuristic.json
```

### Evaluate V3-lite fixed scenarios

```powershell
D:\MiniConda\python.exe -m mahjong_ai.eval.evaluate_scenarios `
  --model artifacts/checkpoints/ppo_v3_lite_action_value_bc_finetune/periodic/model_XXXX_steps.zip `
  --config configs/ppo_v3_lite_action_value_bc_finetune.yaml `
  --output artifacts/reports/scenarios_v3_lite_bc_finetune_XXXX.txt `
  --json-output artifacts/reports/scenarios_v3_lite_bc_finetune_XXXX.json
```

### V3-lite checkpoints to inspect

建议优先检查：

```text
BC accuracy 是否接近/超过 0.70
300w PPO: scenarios 是否明显超过 V2.8 的 12/18
800w PPO: avg_score 是否优于 V2.8 同步数
claim_pass_rate 是否不再长期为 0
discard_regression_rate 是否低于 0.14
```

## V3-full action scorer

V3-full 是真正的逐动作打分模型，不再把所有动作特征简单拼接给 MLP。结构是：

```text
state -> StateEncoder -> state_embedding
action_features[action] -> ActionEncoder -> action_embedding
concat(state_embedding, action_embedding) -> ActionScorer -> action_logit
```

V3-full 使用 78 维 action features，包含 action/tile identity、吃/杠子类型、suit/rank、手牌张数、是否拆对子/面子/搭子等信息。它不能复用 V3-lite 的 18 维 BC dataset，需要重新采集数据。

配置文件：

```text
configs/bc_v3_full_action_scorer.yaml
configs/ppo_v3_full_action_scorer_bc_finetune.yaml
configs/ppo_v3_full_action_scorer_scratch.yaml
```

### 1. Collect V3-full heuristic dataset

```powershell
D:\MiniConda\python.exe -m mahjong_ai.imitation.collect_heuristic_dataset `
  --config configs/bc_v3_full_action_scorer.yaml
```

### 2. Train V3-full BC warmup model

```powershell
D:\MiniConda\python.exe -m mahjong_ai.imitation.train_behavior_clone `
  --config configs/bc_v3_full_action_scorer.yaml
```

输出模型：

```text
artifacts/checkpoints/v3_full_action_scorer_bc/bc_model.zip
```

### 3. PPO finetune from V3-full BC model

```powershell
D:\MiniConda\python.exe -m mahjong_ai.train.train_ppo `
  --config configs/ppo_v3_full_action_scorer_bc_finetune.yaml `
  --resume artifacts/checkpoints/v3_full_action_scorer_bc/bc_model.zip `
  --reset-timesteps
```

### Alternative: Train V3-full PPO from scratch

```powershell
D:\MiniConda\python.exe -m mahjong_ai.train.train_ppo `
  --config configs/ppo_v3_full_action_scorer_scratch.yaml
```

### Evaluate V3-full BC finetune checkpoint

把 `XXXX` 替换成实际 checkpoint 步数，例如 `999996`。

```powershell
D:\MiniConda\python.exe -m mahjong_ai.eval.evaluate `
  --model artifacts/checkpoints/ppo_v3_full_action_scorer_bc_finetune/periodic/model_XXXX_steps.zip `
  --config configs/ppo_v3_full_action_scorer_bc_finetune.yaml `
  --num-games 5000 `
  --opponent heuristic `
  --output artifacts/reports/eval_v3_full_action_scorer_XXXX_heuristic.json
```

### Evaluate V3-full fixed scenarios

```powershell
D:\MiniConda\python.exe -m mahjong_ai.eval.evaluate_scenarios `
  --model artifacts/checkpoints/ppo_v3_full_action_scorer_bc_finetune/periodic/model_XXXX_steps.zip `
  --config configs/ppo_v3_full_action_scorer_bc_finetune.yaml `
  --output artifacts/reports/scenarios_v3_full_action_scorer_XXXX.txt `
  --json-output artifacts/reports/scenarios_v3_full_action_scorer_XXXX.json
```

### Play one readable V3-full game

```powershell
D:\MiniConda\python.exe -m mahjong_ai.eval.play_game `
  --model artifacts/checkpoints/ppo_v3_full_action_scorer_bc_finetune/periodic/model_XXXX_steps.zip `
  --config configs/ppo_v3_full_action_scorer_bc_finetune.yaml `
  --seed 20260523 `
  --opponent heuristic `
  --output artifacts/reports/game_v3_full_action_scorer_XXXX_vs_heuristic.txt `
  --json-output artifacts/reports/game_v3_full_action_scorer_XXXX_vs_heuristic.json
```

### V3-full checkpoints to inspect

优先检查：

```text
BC accuracy 是否接近 V3-lite 的 0.86
100w PPO avg_score 是否保持正数
discard_regression_rate 是否继续低于 0.03
claim_pass_rate 是否高于 V3-lite 100w 的 0.042
kong_rate 是否低于 V3-lite 100w 的 0.0298
model_latency_ms 是否仍可接受
```

### V3 history Transformer from scratch, no old model opponents

```powershell
python -m mahjong_ai.train.train_ppo `
  --config configs/ppo_v3_history_scratch_foundation.yaml
```

### Evaluate V3 scratch checkpoint

```powershell
python -m mahjong_ai.eval.evaluate `
  --model artifacts/checkpoints/ppo_v3_history_scratch_foundation/periodic/model_XXXX_steps.zip `
  --num-games 5000 `
  --opponent heuristic `
  --output artifacts/reports/eval_v3_scratch_XXXX_heuristic.json
```

### V2.6 continue from the 1900w model

```powershell
python -m mahjong_ai.train.train_ppo `
  --config configs/ppo_v26_overnight_from_1900w.yaml `
  --resume artifacts/checkpoints/ppo_v2_human_finetune_high_model_gpu_opponents_stable/periodic/model_19000000_steps.zip
```

### Evaluate V2.6 checkpoint

```powershell
python -m mahjong_ai.eval.evaluate `
  --model artifacts/checkpoints/ppo_v26_overnight_from_1900w/periodic/model_XXXX_steps.zip `
  --num-games 5000 `
  --opponent heuristic `
  --output artifacts/reports/eval_v26_XXXX_heuristic.json
```

### Play one readable V2.6 game

```powershell
python -m mahjong_ai.eval.play_game `
  --model artifacts/checkpoints/ppo_v26_overnight_from_1900w/periodic/model_XXXX_steps.zip `
  --config configs/ppo_v26_overnight_from_1900w.yaml `
  --seed 20260523 `
  --opponent heuristic `
  --output artifacts/reports/game_v26_XXXX_vs_heuristic.txt `
  --json-output artifacts/reports/game_v26_XXXX_vs_heuristic.json
```

### V3 history Transformer from scratch

```powershell
python -m mahjong_ai.train.train_ppo `
  --config configs/ppo_v3_history_transformer_night.yaml
```

### Evaluate V3 checkpoint

```powershell
python -m mahjong_ai.eval.evaluate `
  --model artifacts/checkpoints/ppo_v3_history_transformer_night/periodic/model_XXXX_steps.zip `
  --num-games 5000 `
  --opponent heuristic `
  --output artifacts/reports/eval_v3_history_XXXX_heuristic.json
```
