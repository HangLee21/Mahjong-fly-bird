# 模型优化空间分析（Model Optimization Space）

> 基于最新代码（`af39f04`，2026 合并后）与 `RESULTS.md` / `TRAINING_RUNBOOK.md` 实验数据整理。
> 当前基线：BC(321 人类轨迹, 10ep) 初始化 + TableAttention PPO 对 heuristic 微调，
> 210k 步时 `win_rate≈0.17`（heuristic 基线 ≈0.165，噪声大）、`avg_score≈-0.56`、
> `deal_in≈0.10`、`kong_rate=0`。

## 0. 一句话结论

模型在"合法出牌 / 不无脑杠 / 节奏快"上已经达标，但**强度只到 heuristic 基线水平**，
瓶颈依次是：**人类数据量 → 训练吞吐（步数）→ 奖励与课程的精细度 → 架构信息量**。
按这个顺序投入，收益最大、风险最低。

---

## 1. 数据（最大杠杆，当前最稀缺）

现状：

- 只有 **321 条人类轨迹**，纯 BC 模型对 heuristic 胜率 0%（`RESULTS.md`）。
- 线上已把已结束房间保留 7 天（`ROOM_FINISHED_TTL_MS`），数据会持续增长，但目前导出
  轨迹**不含动作价值特征**（`TRAINING_RUNBOOK.md` §4 明确说明）。

优化空间：

| 项 | 说明 | 预估收益 |
| --- | --- | --- |
| 持续沉淀人类牌谱 | 数据翻倍 → BC 质量线性改善 | 高 |
| 玩家质量过滤/加权 | 按段位/胜率加权 BC（计划文档 §16.2） | 中 |
| 数据增强：座位轮换 | 4 个座位的公共信息对称，1 条轨迹可派生成 4 条 | 高（4x 数据） |
| 数据增强：花色轮换 | 万/筒/条三花色同构，可再 ×3 | 高（3x 数据） |
| 导出脚本补动作价值特征 | `export_human_traces.py` 需要按 `action_features` 配置重建 128×18 特征，否则 BC 与 PPO 观测不一致 | 必要项 |
| 终局标签回填 | 用终局结果给中间步打危险/听牌标签（Risk/Belief 辅助任务数据） | 中（为 V6 铺路） |

> 注意：BC 观测必须与 PPO 观测**逐位对齐**，否则 BC 初始化收益归零。

## 2. 训练吞吐与规模（第二杠杆）

现状：

- 本机 RTX 4060 Laptop，单卡前台训练，330k 步即平台期（`ep_rew` 还在涨但胜率不动）。
- 线上 v3 是 **3000 万步**训练出来的；本地只有几十万步。
- 动作价值特征让 env 步进慢约 **10 倍**（每个合法动作都要 `rule_adapter.step` 模拟）。

优化空间：

| 项 | 说明 |
| --- | --- |
| 并行环境 | `subproc` + `num_envs 8-16`、`n_steps 4096`、`batch_size 8192`（runbook §9） |
| 换更好的机器 | Linux 多核多卡；Windows 下 subproc 可能卡死（上游已默认 dummy 规避） |
| 学习率调度 | BC 微调 `lr=3e-6`，纯 PPO `1e-4`；`target_kl≈0.035` 早停保护 |
| 训练量 | 目标至少数百万步；用新增的 **PPO 续训**（`--resume`，上游已支持）跑长程 |
| 动作特征计算优化 | 每步对所有合法动作做完整模拟是 10x 瓶颈；可只对"候选动作"模拟、或对相同手牌缓存特征；`effective_tiles: false` 可省一部分计算 |
| 评估降噪 | 100-300 局评估噪声大（0.17 vs 0.165 差距不可靠）；评估加到 1000+ 局、多 seed |

## 3. 模型架构

现状（合并后 feature_extractor.py 有 4 个提取器）：

- `LayerNormMLPExtractor`：纯 MLP。
- `HybridHistoryTransformerExtractor`：静态 + 动作历史。
- `TableAttentionTransformerExtractor`（上游新增）：静态 + 4 座位牌面注意力 +（可选）历史。
- `HybridHistoryTransformerV2Extractor`（本地 WIP）：历史 + CLS 掩码注意力，更大 `d_model=192/3 层`。

优化空间：

| 项 | 说明 |
| --- | --- |
| 动作价值特征是核心 | 2698→2826 维静态观测是 v3 超过 heuristic 的主因，**不要关闭**（runbook §10 明确警告） |
| 新提取器消融 | TableAttention vs HistoryV2 只跑一个即可（当前两条线重复）；优先保 TableAttention + 历史（`has_history` 路径） |
| 动作特征维度 | `ACTION_FEATURE_DIM=18` vs `ACTION_FEATURE_FULL_DIM=78`：full 版含牌面 one-hot，对共享打分器必要，但内存/速度翻 4 倍，按目标消融 |
| 共享 ActionValuePolicy（本地 WIP） | `action_value_policy.py`（未提交）：策略输入动作特征而非展开矩阵，可显著省参数/加速，值得并入主线验证 |
| Risk/Belief 头（计划 V6） | 尚未实现；危险度/听牌概率辅助任务可降放铳率（当前 deal_in≈0.10-0.17 仍有空间） |
| 观测剪枝 | 当前 2826 维里动作特征占 2304；若用共享打分器可回到 ~500 维，训练与推理都快 |

## 4. 奖励设计

现状（合并后 reward.py 已同时包含）：

- 上游：`discard_danger_score`（危险牌惩罚）。
- 本地 WIP：打牌效率（向听/牌型/搭子）、价值顺序（孤张字牌）、副露决策、杠判断、手牌目标 shaping。
- 已验证结论（`RESULTS.md`）：**奖励改动 + 续训会退化**（winboost 例）；minimal 稀疏奖励 60k 步学不动；dense shaping 部分被"刷分"（330k 时 ep_rew 涨、胜率平）。

优化空间：

| 项 | 说明 |
| --- | --- |
| 奖励迭代必须全新训练 | 每次改奖励开新 run，不复用旧 checkpoint（否则 value 失配） |
| 奖励密度课程 | 先 dense shaping 学到基础，再逐步调低 shaping 权重、提高终局权重，最后接近稀疏目标 |
| 系数扫描 | 现有 20+ 个 shaping 系数全部叠在 `score_scale=10` 上，可做一轮系数消融（建议固定其它项逐项 0/1 开关） |
| 防"刷分" | `hand_goal` / `ready_bonus` 容易被刷；考虑只对"接近听牌"局面启用，或按局做 reward clipping |
| 终局权重 | 当前 `terminal_win_bonus=0.03` 相对 step_penalty 0.0012 偏弱，可提高自摸/荣和差距 |

## 5. 对手与课程

现状（`RESULTS.md`）：

- win_first 课程有效（210k 最佳 0.17 走的是 heuristic 直训；120k winfirst 也有 0.167）。
- 混合对手池对 heuristic 反而略降（0.10），但鲁棒性更好。
- 本地 WIP 有 `per_seat_sample` + GPU 模型对手池（v1_1000w / v2_1500w 权重 0.45）配置。

优化空间：

| 项 | 说明 |
| --- | --- |
| 阶段课程 | random(10%) → win_first(20%) → heuristic(30%) → 历史模型池(40%)，逐步提高对手强度 |
| 模型池自动扩容 | 训练中每 N 步把当前 checkpoint 加入池（计划文档 V2 §3），对抗固定对手过拟合 |
| 对手多样性 | 池内混入本地 WIP 的 GPU 模型对手（比 CPU 快），注意 `deterministic: false` 保证探索 |
| 评估对手统一 | 上线评估固定用 heuristic + 最近模型双基线，避免课程与评估错位 |

## 6. 算法细节

- BC 辅助损失在 PPO 中**当前有害**（数据太弱），数据增长后可重试（`bc_aux` 开关已在上游）。
- PPO 超参：`clip 0.10-0.15`、`ent 0.01-0.018`、`n_epochs 4-5`、`max_grad_norm 0.5`、`vf_coef 0.5` 已有稳定值，优先不动。
- 多 seed 复现：胜率噪声大，正式结论至少 3 seed 取中位。
- 续训功能已就绪（`--resume` / `--reset-timesteps`），用于**同奖励长程**训练。

## 7. 部署与推理（收尾项）

- `server.py`（本地未提交）已提供推理服务；部署时需把 `observation.py` / `feature_extractor.py` /
  `reward.py` 与后端同步（runbook §11）。
- 推理侧可用 `int8/bf16` 或小模型蒸馏；麻将延迟预算宽松，**非优先**。
- 建议加 BC top-1/top-k 命中率 + 副露率/防守率等"像不像人"指标，防止 PPO 微调后打法漂移。

---

## 8. 建议的下一步（按性价比排序）

1. **数据增强落地**：座位轮换 + 花色轮换，把 321 条轨迹扩到 ~4k 条；导出脚本补动作特征。
2. **换大机器长训**：subproc 8-16 env，BC→PPO heuristic 跑 3M+ 步（当前只跑到 330k）。
3. **奖励微调一轮全新训练**：固定系数做 0/1 消融，验证 danger/efficiency 两项的增量。
4. **架构收敛**：TableAttention（+历史）作为主线，HistoryV2 与共享 ActionValuePolicy 做 A/B，选一个并入。
5. **评估升级**：1000 局 + 3 seed + 多基线，拿到可信的胜率/得分差再做下一个决策。
