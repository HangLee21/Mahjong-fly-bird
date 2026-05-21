# 麻将 AI 模型迭代计划与最终架构设计

> 用途：本文档用于指导麻将 AI 模型从基础强化学习版本，逐步迭代到融合真实人类牌谱、自博弈、对抗式训练、风险预测和对手建模的高级版本。  
> 适用范围：训练侧 `training/` 仓库，也可作为整个项目的模型路线总纲。  
> 核心目标：先实现一个可训练、可部署、可持续迭代的麻将 AI，再逐步利用真实牌局数据提升模型强度和人类风格。

---

## 0. 总体目标

本项目中的麻将 AI 不应设计成一个直接“记规则”的大语言模型，而应设计成：

```text
规则引擎 + 合法动作约束 + 策略价值网络 + 自博弈训练 + 人类牌谱增强
```

核心思想是：

```text
规则引擎负责“什么动作合法”；
模型负责“在合法动作中选哪个更好”；
真实牌谱负责“让模型学会人类打法”；
自博弈负责“让模型超过简单模仿”；
对抗式训练负责“发现模型漏洞并提升鲁棒性”。
```

最终目标不是简单做一个能出牌的 bot，而是构建一个可以持续进化的麻将智能体：

```text
V1：能合法打牌
V2：能稳定击败基础 bot
V3：能模仿真实玩家
V4：能通过自博弈持续变强
V5：具备风险判断和对手建模能力
V6：具备对抗鲁棒性和线上持续学习能力
```

---

## 1. 整体技术路线

整体模型迭代路线如下：

```text
规则引擎
  ↓
MaskablePPO 基础模型
  ↓
启发式 bot / 历史模型池 self-play
  ↓
真实人类牌谱收集
  ↓
行为克隆 Behavior Cloning
  ↓
PPO / Self-play 微调
  ↓
对抗式模仿学习 GAIL
  ↓
Risk / Belief 多任务头
  ↓
线上 A/B 测试与持续迭代
```

推荐的阶段性实现顺序：

```text
阶段 1：MLP + MaskablePPO
阶段 2：历史模型池 self-play
阶段 3：真实牌谱行为克隆
阶段 4：BC 初始化 + PPO 微调
阶段 5：动作历史 Transformer
阶段 6：Risk Head / Belief Head
阶段 7：GAIL / Exploit Opponent
阶段 8：线上持续学习与模型版本管理
```

---

## 2. 当前基础模型：V1 MLP + MaskablePPO

### 2.1 模型定位

V1 是整个系统的最小可运行版本。

它的目标不是最强，而是：

```text
1. 能接收后端 observation；
2. 能处理 action mask；
3. 能输出合法动作；
4. 能通过 PPO 训练；
5. 能导出并部署；
6. 能接入后端 AI 推理服务。
```

### 2.2 模型结构

```text
Observation Vector
    ↓
MLP Feature Extractor
    ↓
Actor-Critic 双头
    ├── Policy Head：输出动作 logits
    └── Value Head：输出局面价值
    ↓
Action Mask
    ↓
Softmax over legal actions
    ↓
Selected Action
```

### 2.3 数学形式

策略网络：

```math
\pi_\theta(a_t \mid o_t)
```

价值网络：

```math
V_\phi(o_t)
```

其中：

```text
o_t：当前玩家可见 observation；
a_t：当前动作；
θ：策略网络参数；
φ：价值网络参数。
```

action mask：

```math
m_t(a)=
\begin{cases}
1, & a \in A_{\text{legal}}(o_t) \\
0, & a \notin A_{\text{legal}}(o_t)
\end{cases}
```

非法动作 logits 置为极小值：

```math
\tilde{z}(a)=
\begin{cases}
z(a), & m_t(a)=1 \\
-\infty, & m_t(a)=0
\end{cases}
```

最终策略：

```math
\pi_\theta(a\mid o_t)
=
\frac{\exp(\tilde{z}(a))}
{\sum_{a'}\exp(\tilde{z}(a'))}
```

### 2.4 训练目标

使用 PPO clipped objective：

```math
L^{CLIP}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
r_t(\theta)A_t,
\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
```

其中：

```math
r_t(\theta)=
\frac{\pi_\theta(a_t\mid o_t)}
{\pi_{\theta_{\text{old}}}(a_t\mid o_t)}
```

### 2.5 V1 验收标准

```text
[ ] 能完成环境 reset / step；
[ ] action mask 正确；
[ ] 非法动作率为 0；
[ ] 能跑 random games；
[ ] 能启动 MaskablePPO 训练；
[ ] 训练后模型能保存和加载；
[ ] predictor 能返回合法动作；
[ ] 后端可以调用模型服务。
```

---

## 3. V2：历史模型池 Self-play

### 3.1 为什么需要 self-play

如果模型只和固定启发式 bot 对战，容易出现：

```text
1. 只会针对某一种固定对手；
2. 策略泛化差；
3. 遇到新打法容易崩；
4. 可能学到利用 bot 缺陷的偏门策略。
```

因此需要引入历史模型池。

### 3.2 对手池设计

维护一个 opponent pool：

```text
opponent_pool/
├── random_agent
├── heuristic_agent
├── shanten_agent
├── model_v1
├── model_v2
├── model_v3
└── human_cloned_model
```

训练时随机采样对手：

```text
50% 历史模型；
20% 启发式 bot；
20% 最近模型；
10% 特殊 exploit 模型。
```

### 3.3 训练流程

```text
1. 使用 V1 模型作为初始模型；
2. 当前模型对战 heuristic / random；
3. 得到 model_v2；
4. 将 model_v1 加入 opponent pool；
5. model_v2 对战 model_v1 + heuristic；
6. 得到 model_v3；
7. 持续循环。
```

### 3.4 新模型进入模型池条件

```text
[ ] 非法动作率为 0；
[ ] 对 random agent 明显占优；
[ ] 对 heuristic agent 平均收益不低；
[ ] 对上一版模型不明显劣化；
[ ] 不同座位平均收益差异合理；
[ ] 平均局长无异常；
[ ] 没有明显拖局行为。
```

---

## 4. V3：真实人类牌谱行为克隆

### 4.1 为什么加入真实牌谱

纯自博弈模型可能会出现：

```text
1. 打法不像人；
2. 学到奇怪策略；
3. 在真实玩家面前表现不稳定；
4. 对人类常见套路理解不足。
```

真实牌谱可以让模型学习：

```text
1. 人类常见出牌习惯；
2. 攻守转换；
3. 副露时机；
4. 危险牌规避；
5. 真实玩家的节奏和风格。
```

### 4.2 牌谱数据格式

每一步至少需要保存：

```json
{
  "game_id": "game_001",
  "step": 35,
  "player_id": 2,
  "observation": [0.0, 1.0],
  "legal_actions": [1, 5, 100],
  "action": 5,
  "final_score": 12,
  "action_source": "human",
  "player_level": "high",
  "model_version": null
}
```

必须保留：

```text
observation；
legal_actions；
human action；
final score；
player level；
是否托管；
是否断线；
是否完整对局。
```

### 4.3 牌谱清洗

不能把所有牌谱直接用于训练。

需要剔除：

```text
断线局；
托管局；
未完整结束的局；
明显乱打局；
规则版本不一致的局；
observation 版本不一致的局；
action 不在 legal_actions 中的异常样本。
```

建议保留：

```text
高胜率玩家；
高段位玩家；
完整对局；
高质量回放；
人类真实操作而非 AI 托管操作。
```

### 4.4 行为克隆目标

行为克隆损失：

```math
L_{BC}(\theta)
=
-\sum_t \log \pi_\theta(a_t^{human}\mid o_t)
```

含义：

```text
在人类玩家当时看到的局面下，让模型提高人类实际选择动作的概率。
```

如果有玩家质量权重：

```math
L_{BC}(\theta)
=
-\sum_t w_t \log \pi_\theta(a_t^{human}\mid o_t)
```

其中：

```text
w_t：该样本权重，高水平玩家权重更高。
```

### 4.5 V3 训练流程

```text
1. 收集线上人类牌谱；
2. 清洗并筛选高质量样本；
3. 生成 supervised dataset；
4. 用当前 policy head 做行为克隆；
5. 得到 human_cloned_model；
6. 与 V2 self-play 模型对比评估。
```

### 4.6 V3 验收指标

```text
[ ] 行为克隆准确率高于随机；
[ ] top-k action 命中率合理；
[ ] 模型动作不违反 legal_actions；
[ ] 模型打法更接近真实玩家；
[ ] 与启发式 bot 对战不明显退化；
[ ] 可作为 PPO 微调初始模型。
```

---

## 5. V4：BC 初始化 + PPO 微调

### 5.1 为什么不能只做行为克隆

行为克隆只能模仿人类，不能保证长期最优。

它的问题是：

```text
1. 会继承人类错误；
2. 对罕见局面泛化差；
3. 无法主动探索更优策略；
4. 只能学“人类做了什么”，不能学“这样做最终收益如何”。
```

因此行为克隆后需要继续 PPO 微调。

### 5.2 训练流程

```text
人类牌谱
  ↓
Behavior Cloning
  ↓
human_cloned_model
  ↓
PPO / self-play fine-tuning
  ↓
human_enhanced_rl_model
```

### 5.3 混合目标

训练中可使用：

```math
L =
L_{PPO}
+
\alpha L_{BC}
+
\beta L_{value}
+
\delta L_{entropy}
```

其中：

```text
L_PPO：强化学习策略损失；
L_BC：行为克隆损失；
L_value：价值函数损失；
L_entropy：探索正则；
α、β、δ：权重。
```

推荐训练权重策略：

```text
早期：
  α 较大，保持人类风格；
  PPO 学习率较小。

中期：
  α 逐渐降低；
  PPO 权重提高。

后期：
  PPO 主导；
  BC 作为轻量正则。
```

### 5.4 V4 验收标准

```text
[ ] 模型不弱于纯 BC 模型；
[ ] 模型不弱于 V2 self-play 模型；
[ ] 真实玩家牌谱上的动作相似度不过度下降；
[ ] 平均得分提升；
[ ] 放铳率不异常升高；
[ ] 对历史模型池表现稳定。
```

---

## 6. V5：动作历史 Transformer

### 6.1 为什么需要历史编码

麻将决策高度依赖历史动作。

例如：

```text
某个玩家连续打中张，可能牌型已接近完成；
某个玩家早早碰牌，可能进攻意图强；
某张牌一直没人打，后期可能危险；
某个玩家多次过牌，可能不需要某类牌。
```

单纯 observation 计数可能丢失动作顺序信息。

### 6.2 历史动作输入

每一步历史动作可以编码为：

```text
actor_id；
action_type；
tile_id；
step_index；
relative_turn；
是否副露；
是否响应动作；
当前分数变化。
```

形成序列：

```math
H_t = \{h_1, h_2, ..., h_t\}
```

使用 Transformer 或 GRU 编码：

```math
z_t = \text{HistoryEncoder}(H_t)
```

### 6.3 V5 模型结构

```text
Current Observation Encoder
        ↓
current_feature
        ↓
History Transformer
        ↓
history_feature
        ↓
Feature Fusion
        ↓
Policy Head + Value Head
        ↓
Action Mask
        ↓
Final Action
```

### 6.4 什么时候上 Transformer

不要在第一版就上 Transformer。

推荐条件：

```text
[ ] 规则环境稳定；
[ ] V1/V2/V3/V4 已经跑通；
[ ] 有足够自博弈数据；
[ ] 有足够真实牌谱；
[ ] 线上推理延迟预算允许。
```

---

## 7. V6：Risk Head 与 Belief Head

### 7.1 为什么需要风险与信念建模

麻将是不完全信息博弈。  
模型不能看到对手暗手牌，但可以根据公开信息估计：

```text
对手是否听牌；
某张牌是否危险；
对手可能需要什么牌；
隐藏牌大致分布；
当前应该进攻还是防守。
```

这些信息对最终策略非常关键。

### 7.2 Risk Head

Risk Head 预测：

```text
打出每张牌的危险度；
当前放铳风险；
每个对手的进攻状态。
```

输出示例：

```text
tile_risk_scores: shape = [N_TILE_TYPES]
opponent_attack_prob: shape = [3]
deal_in_prob: scalar
```

数学表示：

```math
R_\omega(o_t, tile)
```

表示当前打出某张牌的风险。

### 7.3 Belief Head

Belief Head 预测隐藏信息的概率分布：

```text
对手可能持有哪些牌；
剩余牌分布；
对手听牌概率；
对手可能胡哪些牌。
```

数学表示：

```math
B_\eta(hidden \mid o_t)
```

### 7.4 辅助损失

如果能从牌谱终局还原部分标签，可以训练：

```math
L_{risk}
=
\text{BCE}(risk\_pred, risk\_label)
```

```math
L_{belief}
=
\text{CE}(belief\_pred, hidden\_label)
```

总损失变为：

```math
L =
L_{PPO}
+
\alpha L_{BC}
+
\beta L_{value}
+
\gamma L_{risk}
+
\lambda L_{belief}
+
\delta L_{entropy}
```

### 7.5 V6 验收标准

```text
[ ] 放铳率下降；
[ ] 平均收益不下降；
[ ] 风险预测有校准能力；
[ ] 高风险牌预测与真实放铳事件相关；
[ ] 防守局面下动作更合理；
[ ] 对强启发式和历史模型池表现更稳。
```

---

## 8. V7：GAIL 对抗式模仿学习

### 8.1 为什么引入 GAIL

行为克隆只在单步动作上模仿人类：

```text
这个局面下，人类打了什么。
```

但它不直接学习整段轨迹是否像高水平人类。

GAIL 引入判别器：

```text
Discriminator 判断一段 observation-action 是否像人类；
Policy 试图生成让判别器认为像人类的动作轨迹。
```

### 8.2 判别器

判别器：

```math
D_\psi(o_t, a_t)
```

输出：

```text
当前 observation-action 来自人类专家的概率。
```

### 8.3 GAIL 奖励

可以定义模仿奖励：

```math
r_t^{imit}
=
-\log(1-D_\psi(o_t,a_t))
```

最终奖励：

```math
r_t
=
r_t^{game}
+
\lambda r_t^{imit}
```

其中：

```text
r_game：真实牌局收益；
r_imit：像不像人类专家的奖励；
λ：模仿奖励权重。
```

### 8.4 使用注意

GAIL 不应太早引入。

原因：

```text
1. 实现复杂；
2. 对数据质量敏感；
3. 容易过度模仿人类；
4. 可能牺牲长期收益；
5. 判别器过强或过弱都会影响训练。
```

推荐在已有较好 V4/V5/V6 模型后再尝试。

---

## 9. V8：Exploit Opponent 对抗评估

### 9.1 思路

专门训练一个 exploit opponent，用来克制当前模型。

```text
固定当前主模型；
训练一个对手模型；
目标是最大化对主模型的收益；
找到主模型弱点。
```

这不是上线模型，而是测试模型漏洞的工具。

### 9.2 使用方式

```text
1. 冻结 main_model；
2. 训练 exploit_model 对战 main_model；
3. 如果 exploit_model 能显著获利，说明 main_model 有漏洞；
4. 将 exploit_model 加入 opponent pool；
5. 用 main_model 继续对抗训练。
```

### 9.3 作用

```text
发现固定策略漏洞；
提升鲁棒性；
防止模型被某种打法克制；
增强线上真实玩家环境下的稳定性。
```

---

## 10. 最终高级模型架构

最终推荐模型结构如下：

```text
                           ┌─────────────────────┐
                           │ Current Observation │
                           └──────────┬──────────┘
                                      ↓
                           ┌─────────────────────┐
                           │ Observation Encoder │
                           └──────────┬──────────┘

                           ┌─────────────────────┐
                           │  Action History Seq │
                           └──────────┬──────────┘
                                      ↓
                           ┌─────────────────────┐
                           │ History Transformer │
                           └──────────┬──────────┘

                           ┌─────────────────────┐
                           │ Public State Feature│
                           └──────────┬──────────┘
                                      ↓
                           ┌─────────────────────┐
                           │    Feature Fusion   │
                           └──────────┬──────────┘
                                      ↓
        ┌──────────────────────┬──────────────────────┬──────────────────────┐
        ↓                      ↓                      ↓                      ↓
┌───────────────┐      ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│ Policy Head   │      │ Value Head    │      │ Risk Head     │      │ Belief Head   │
│ action logits │      │ state value   │      │ tile danger   │      │ hidden state  │
└───────┬───────┘      └───────────────┘      └───────────────┘      └───────────────┘
        ↓
┌───────────────┐
│ Action Mask   │
└───────┬───────┘
        ↓
┌───────────────┐
│ Final Action  │
└───────────────┘
```

### 10.1 输入模块

```text
1. 当前手牌；
2. 自己副露；
3. 所有玩家弃牌；
4. 所有玩家明牌；
5. 分数；
6. 庄家；
7. 当前玩家；
8. 上一张弃牌；
9. 动作历史；
10. 当前 legal actions。
```

### 10.2 输出模块

```text
Policy Head：
  输出动作 logits，用于最终决策。

Value Head：
  输出当前局面预期收益，用于 PPO。

Risk Head：
  输出放铳风险、危险牌估计，用于辅助训练。

Belief Head：
  输出隐藏信息估计，例如对手听牌概率、隐藏牌分布。

Imitation / Discriminator：
  可选，用于 GAIL 或人类风格约束。
```

---

## 11. 最终训练目标设计

总损失可以写成：

```math
L_{\text{total}}
=
L_{\text{PPO}}
+
\alpha L_{\text{BC}}
+
\beta L_{\text{value}}
+
\gamma L_{\text{risk}}
+
\lambda L_{\text{belief}}
+
\delta L_{\text{entropy}}
+
\eta L_{\text{GAIL}}
```

各项含义：

```text
L_PPO：
  强化学习主目标，提升长期收益。

L_BC：
  行为克隆损失，让模型学习真实人类打法。

L_value：
  价值函数损失，提高局面评估能力。

L_risk：
  风险预测损失，提升防守能力。

L_belief：
  隐藏信息估计损失，提升对手建模能力。

L_entropy：
  熵正则，避免策略过早收敛。

L_GAIL：
  对抗式模仿损失，让模型轨迹接近高水平人类。
```

不同阶段权重不同：

```text
早期：
  L_BC 权重大；
  L_PPO 较小；
  目标是学会像人类。

中期：
  L_PPO 权重提升；
  L_BC 降低；
  目标是提升收益。

后期：
  L_PPO 主导；
  L_risk / L_belief / L_GAIL 辅助；
  目标是增强鲁棒性和高级判断。
```

---

## 12. 数据闭环设计

线上小程序需要持续沉淀训练数据。

### 12.1 每一步保存

```json
{
  "game_id": "game_001",
  "step": 35,
  "player_id": 2,
  "observation": [0.0, 1.0],
  "legal_actions": [1, 5, 100],
  "action": 5,
  "action_source": "human",
  "model_version": null,
  "state_hash_before": "abc",
  "state_hash_after": "def",
  "reward": 0.0
}
```

### 12.2 每局保存

```json
{
  "game_id": "game_001",
  "rule_version": "rule_v1",
  "observation_version": "obs_v1",
  "action_version": "action_v1",
  "players": [],
  "final_scores": [12, -3, -4, -5],
  "winner": 0,
  "draw": false,
  "total_steps": 83
}
```

### 12.3 数据用途

```text
行为克隆；
离线评估；
风险标签构造；
隐藏信息建模；
失败局面回放；
模型 A/B 测试；
线上策略漏洞分析。
```

---

## 13. 模型版本规划

### V1：基础强化学习版

```text
MLP Actor-Critic
+ Action Mask
+ MaskablePPO
+ Heuristic Opponents
```

目标：

```text
能训练、能部署、能合法出牌。
```

### V2：历史模型池版

```text
V1
+ Opponent Pool
+ Historical Self-play
```

目标：

```text
减少对固定 bot 过拟合，提升泛化。
```

### V3：人类牌谱模仿版

```text
V2
+ Behavior Cloning
+ Human Replay Dataset
```

目标：

```text
让模型学会真实玩家打法。
```

### V4：人类增强强化学习版

```text
V3
+ PPO Fine-tuning
+ BC Regularization
```

目标：

```text
既像人，又追求长期收益。
```

### V5：历史序列增强版

```text
V4
+ Action History Transformer
```

目标：

```text
理解牌局节奏和动作顺序。
```

### V6：风险与信念建模块版

```text
V5
+ Risk Head
+ Belief Head
```

目标：

```text
提升防守、危险牌判断和对手建模。
```

### V7：对抗式模仿版

```text
V6
+ GAIL Discriminator
+ Human-style Reward
```

目标：

```text
让模型轨迹更接近高水平人类。
```

### V8：对抗评估与鲁棒版

```text
V7
+ Exploit Opponent
+ Robust Self-play
```

目标：

```text
发现并修复模型漏洞，提高线上鲁棒性。
```

---

## 14. 推荐落地顺序

实际开发中，不建议一开始就实现最终架构。

推荐顺序是：

```text
第 1 步：实现规则引擎和训练环境；
第 2 步：实现 V1 MLP + MaskablePPO；
第 3 步：部署到后端，完成 AI 对局；
第 4 步：上线小程序，收集真实牌谱；
第 5 步：清洗人类牌谱，训练 V3 BC 模型；
第 6 步：用 BC 模型初始化 PPO，训练 V4；
第 7 步：维护历史模型池，训练 V2/V4 self-play；
第 8 步：引入动作历史 Transformer；
第 9 步：引入 Risk / Belief 辅助任务；
第 10 步：引入 GAIL 和 exploit opponent；
第 11 步：线上 A/B 测试和持续迭代。
```

---

## 15. 阶段性验收指标

### 15.1 基础合法性指标

```text
非法动作率；
对局完成率；
环境崩溃率；
状态 hash 一致性；
后端与训练侧 legal_actions 一致性。
```

### 15.2 策略强度指标

```text
平均得分；
胜率 / 和牌率；
放铳率；
听牌率；
流局率；
不同座位平均收益；
对 random / heuristic / historical model 的表现。
```

### 15.3 人类风格指标

```text
行为克隆 top-1 accuracy；
行为克隆 top-k accuracy；
与高水平人类动作分布的 KL divergence；
人类评测主观合理性；
副露率、进攻率、防守率与人类分布差异。
```

### 15.4 鲁棒性指标

```text
对 exploit opponent 的收益；
对不同风格 bot 的收益；
对异常局面表现；
长时间线上运行稳定性；
AI 服务超时 fallback 次数；
模型版本回滚次数。
```

---

## 16. 训练数据质量控制

### 16.1 需要剔除的数据

```text
断线托管局；
规则版本不一致局；
observation 编码版本不一致局；
缺失 legal_actions 的局；
action 不在 legal_actions 中的样本；
明显异常结算局；
测试 bot 局；
短时间重复刷局。
```

### 16.2 样本加权

建议按玩家水平和牌局质量加权：

```text
高水平玩家：权重大；
完整对局：权重大；
托管行为：权重为 0；
低质量对局：权重降低；
AI 自己的对局：单独标记，不混入人类 BC 数据。
```

加权行为克隆：

```math
L_{BC}
=
-\sum_t w_t \log \pi_\theta(a_t^{human}\mid o_t)
```

---

## 17. 最终一句话总结

本项目的最终模型应设计为：

```text
以规则引擎为硬约束，
以 action mask 保证合法性，
以 MLP / Transformer 编码当前局面和历史动作，
以 policy head 输出动作分布，
以 value head 评估局面价值，
以真实人类牌谱进行行为克隆预训练，
以 PPO self-play 优化长期收益，
以 risk / belief head 强化防守和对手建模，
以 GAIL 和 exploit opponent 增强人类风格与对抗鲁棒性。
```

实际落地时应遵循：

```text
先跑通闭环，再提升强度；
先保证合法，再追求智能；
先利用规则环境，再利用真实牌谱；
先模仿人类，再通过自博弈超过人类；
先做可部署模型，再做复杂高级架构。
```
