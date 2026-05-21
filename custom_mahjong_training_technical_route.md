# 训练侧技术路线文档：Custom Mahjong AI Training

> 用途：本文件用于指导 Codex 在 `training/` 文件夹中实现麻将 AI 训练侧代码。  
> 目标：基于已有非标准麻将规则，构建可训练、可评估、可导出、可接入后端推理服务的 AI 模型训练系统。  
> 范围：本文件只覆盖训练侧，不覆盖微信小程序前端实现，也不覆盖完整后端房间系统。

---

## 0. 项目目标

本训练侧项目需要完成以下目标：

```text
1. 接入或复用麻将规则引擎；
2. 将规则引擎封装为强化学习环境；
3. 生成模型可用的 observation；
4. 生成每一步合法动作 action mask；
5. 实现随机与启发式 baseline；
6. 使用 PPO / MaskablePPO 训练第一版模型；
7. 支持 self-play 或历史模型池迭代；
8. 对模型进行离线评估；
9. 导出模型供后端推理服务调用；
10. 固化对局日志格式，为后续真实牌谱再训练做准备。
```

核心原则：

```text
规则判断不交给神经网络；
模型只负责在合法动作集合中选择动作；
训练环境和线上后端必须使用同一套规则语义；
小程序端不参与训练；
第一版优先跑通闭环，不追求最强模型。
```

---

## 1. 总体技术路线

训练侧整体路线如下：

```text
规则说明 / 已实现规则引擎
        ↓
规则适配层 RuleAdapter
        ↓
训练环境 MahjongEnv
        ↓
observation 编码 + action mask
        ↓
baseline agent 验证环境
        ↓
MaskablePPO 训练
        ↓
模型评估 tournament
        ↓
模型导出 TorchScript / ONNX
        ↓
后端 AI 推理服务加载模型
```

第一阶段建议走：

```text
Gymnasium 单智能体环境
+ 其他 3 家使用启发式 bot
+ MaskablePPO
```

第二阶段再升级：

```text
PettingZoo AEC 多智能体环境
+ self-play
+ 历史模型池 opponent pool
```

这样可以降低第一版工程复杂度。

---

## 2. 训练侧推荐目录结构

Codex 应在训练侧文件夹中实现如下结构：

```text
training/
├── README.md
├── requirements.txt
├── configs/
│   ├── env.yaml
│   ├── ppo_debug.yaml
│   ├── ppo_small.yaml
│   ├── ppo_selfplay.yaml
│   └── export.yaml
├── mahjong_ai/
│   ├── __init__.py
│   ├── rules/
│   │   ├── adapter.py
│   │   └── mock_rule_engine.py
│   ├── env/
│   │   ├── gym_env.py
│   │   ├── pettingzoo_env.py
│   │   ├── observation.py
│   │   ├── actions.py
│   │   ├── reward.py
│   │   └── wrappers.py
│   ├── agents/
│   │   ├── base.py
│   │   ├── random_agent.py
│   │   ├── heuristic_agent.py
│   │   └── model_agent.py
│   ├── models/
│   │   ├── mlp_policy.py
│   │   └── feature_extractor.py
│   ├── train/
│   │   ├── train_ppo.py
│   │   ├── self_play.py
│   │   └── callbacks.py
│   ├── eval/
│   │   ├── evaluate.py
│   │   ├── tournament.py
│   │   └── metrics.py
│   ├── export/
│   │   ├── export_onnx.py
│   │   ├── export_torchscript.py
│   │   └── validate_export.py
│   ├── inference/
│   │   ├── predictor.py
│   │   └── schema.py
│   └── utils/
│       ├── config.py
│       ├── seed.py
│       ├── logger.py
│       └── replay.py
├── scripts/
│   ├── check_env.py
│   ├── run_random_games.py
│   ├── train_debug.sh
│   ├── train_ppo.sh
│   ├── evaluate.sh
│   └── export.sh
├── tests/
│   ├── test_rule_adapter.py
│   ├── test_actions.py
│   ├── test_observation.py
│   ├── test_reward.py
│   ├── test_gym_env.py
│   └── test_inference.py
└── artifacts/
    ├── checkpoints/
    ├── exported/
    ├── logs/
    └── reports/
```

---

## 3. 关键设计原则

### 3.1 规则引擎独立于模型

训练侧必须假设存在一个确定性规则引擎。

模型不能判断：

```text
是否能吃；
是否能碰；
是否能杠；
是否能胡；
番型是否成立；
结算是否正确；
动作优先级如何处理。
```

这些都必须由规则引擎负责。

模型只做一件事：

```text
输入 observation + legal_action_mask
输出 action
```

### 3.2 模型不能看到隐藏信息

麻将是不完全信息博弈。  
模型输入中不能包含：

```text
其他玩家暗手牌；
牌墙未来顺序；
未公开的摸牌序列；
任何后验结算信息；
任何未来动作。
```

模型可以看到：

```text
自己的手牌；
自己的副露；
全体玩家弃牌；
全体玩家明牌；
当前分数；
剩余牌数；
庄家位置；
当前行动玩家；
上一个动作；
上一次打出的牌；
当前合法动作 mask。
```

### 3.3 每一步必须使用 action mask

麻将动作空间中，绝大多数动作在某一时刻都是非法动作。  
如果不使用 action mask，模型会频繁探索非法动作，训练效率会很差。

训练环境必须提供：

```python
def action_masks(self) -> np.ndarray:
    ...
```

并保证返回值为：

```text
shape = (ACTION_SPACE_SIZE,)
dtype = bool 或 int8
mask[i] = 1 表示动作 i 当前合法
mask[i] = 0 表示动作 i 当前非法
```

### 3.4 训练环境与后端规则必须一致

训练侧可以先用 mock rule engine 跑通代码结构，但最终必须接入真实规则引擎。

至少需要保证：

```text
同一个初始状态；
同一个动作序列；
训练环境和后端输出相同 legal actions；
训练环境和后端输出相同 next state；
训练环境和后端输出相同 terminal / score。
```

---

## 4. 规则适配层 RuleAdapter

由于训练侧不应该直接依赖后端复杂结构，需要实现一个规则适配层。

文件：

```text
mahjong_ai/rules/adapter.py
```

接口：

```python
from typing import Protocol, Any


class RuleAdapter(Protocol):
    def reset(self, seed: int | None = None) -> Any:
        """创建一局新游戏，返回内部完整 GameState。"""

    def clone_state(self, state: Any) -> Any:
        """深拷贝状态，用于回放、模拟、测试。"""

    def get_current_player(self, state: Any) -> int:
        """返回当前需要行动的玩家 id。"""

    def get_legal_actions(self, state: Any, player_id: int) -> list[int]:
        """返回当前玩家所有合法动作编码。"""

    def step(self, state: Any, player_id: int, action: int) -> Any:
        """执行动作并返回新状态。"""

    def is_terminal(self, state: Any) -> bool:
        """判断当前牌局是否结束。"""

    def get_scores(self, state: Any) -> list[float]:
        """返回当前或终局分数。"""

    def get_winner(self, state: Any) -> int | None:
        """返回赢家 id；流局或未结束返回 None。"""

    def get_public_info(self, state: Any) -> dict:
        """返回所有玩家可见的公开信息。"""

    def get_private_info(self, state: Any, player_id: int) -> dict:
        """返回 player_id 自己可见的私有信息，例如自己的手牌。"""

    def get_state_hash(self, state: Any) -> str:
        """返回状态 hash，用于日志、回放和一致性测试。"""
```

Codex 需要实现：

```text
1. RuleAdapter 协议；
2. MockRuleEngine，用于没有真实规则时跑通训练代码；
3. 后续预留 RealRuleAdapter，用于接入后端真实规则。
```

---

## 5. 动作空间设计

文件：

```text
mahjong_ai/env/actions.py
```

### 5.1 统一动作编码

动作空间应该是固定长度离散空间，便于 PPO 训练。

假设牌型数量为：

```python
N_TILE_TYPES = 34
```

动作编码建议如下：

```python
DISCARD_OFFSET = 0                     # 0 ~ N_TILE_TYPES-1 表示打出某张牌
ACTION_PASS = 100
ACTION_WIN = 101
ACTION_PONG = 102
ACTION_CHOW_LEFT = 103
ACTION_CHOW_MIDDLE = 104
ACTION_CHOW_RIGHT = 105
ACTION_KONG_EXPOSED = 106
ACTION_KONG_CONCEALED = 107
ACTION_KONG_ADDED = 108

ACTION_SPACE_SIZE = 128
```

如果自定义麻将有更多动作，则统一扩展 `ACTION_SPACE_SIZE`。

### 5.2 动作对象与整数编码互转

Codex 应实现：

```python
@dataclass(frozen=True)
class MahjongAction:
    type: str
    tile: int | None = None
    extra: dict | None = None
```

并实现：

```python
def encode_action(action: MahjongAction) -> int:
    ...

def decode_action(action_id: int) -> MahjongAction:
    ...

def build_action_mask(legal_actions: list[int]) -> np.ndarray:
    ...
```

### 5.3 fallback 策略

模型输出异常时需要 fallback。

```python
def fallback_action(legal_actions: list[int]) -> int:
    if ACTION_WIN in legal_actions:
        return ACTION_WIN
    if ACTION_PASS in legal_actions:
        return ACTION_PASS
    return legal_actions[0]
```

注意：fallback 只用于安全兜底，不能掩盖 action mask 的 bug。训练和测试阶段应记录异常。

---

## 6. Observation 编码设计

文件：

```text
mahjong_ai/env/observation.py
```

### 6.1 Observation 设计目标

Observation 必须满足：

```text
固定长度；
只包含当前玩家可见信息；
能表达手牌、弃牌、明牌、分数、位置、合法动作；
适合 MLP 第一版训练；
后续可扩展为 Transformer 输入。
```

### 6.2 第一版推荐使用向量编码

示例：

```text
hand_counts:             N_TILE_TYPES
self_meld_counts:        N_TILE_TYPES
all_discard_counts:      4 * N_TILE_TYPES
all_open_meld_counts:    4 * N_TILE_TYPES
remaining_tile_counts:   N_TILE_TYPES
scores:                  4
dealer_one_hot:          4
current_player_one_hot:  4
relative_position:       4
last_discard_one_hot:    N_TILE_TYPES + 1
round_info:              若干维
legal_action_mask:       ACTION_SPACE_SIZE
```

合并为：

```python
obs = np.concatenate([...]).astype(np.float32)
```

注意：是否把 `legal_action_mask` 拼进 observation 由配置决定。  
即使不拼进 observation，也必须通过 `action_masks()` 提供给 MaskablePPO。

### 6.3 必须实现的函数

```python
def build_observation(
    rule_adapter: RuleAdapter,
    state: Any,
    player_id: int,
    config: dict,
) -> np.ndarray:
    ...
```

```python
def get_observation_dim(config: dict) -> int:
    ...
```

```python
def validate_observation(obs: np.ndarray, expected_dim: int) -> None:
    ...
```

### 6.4 编码规范

Codex 需要保证：

```text
所有计数变量归一化到 0~1 或保留小整数；
分数变量需要归一化，例如除以初始分；
one-hot 编码必须固定长度；
缺失 last_discard 时使用额外一维 none 标识；
observation 中不能出现 NaN / inf。
```

---

## 7. 奖励函数设计

文件：

```text
mahjong_ai/env/reward.py
```

### 7.1 第一版：终局分数奖励

第一版优先使用真实结算作为 reward：

```python
reward = final_score[player_id] - initial_score[player_id]
```

或者归一化：

```python
reward = (final_score[player_id] - initial_score[player_id]) / score_scale
```

优点：

```text
与真实目标一致；
不会引入太多人为偏见；
便于评估。
```

缺点：

```text
奖励稀疏；
训练较慢。
```

### 7.2 第二版：轻量 shaping reward

后续可以加入小幅过程奖励：

```text
向听数减少：+0.01
进入听牌：+0.03
胡牌：+真实得分
放铳：-真实失分
每步轻微时间成本：-0.001
```

注意：

```text
shaping reward 不能改变真实目标；
最终评估只看真实结算；
不要加入过多人工策略偏见。
```

### 7.3 必须实现

```python
def compute_reward(
    prev_state: Any,
    next_state: Any,
    player_id: int,
    rule_adapter: RuleAdapter,
    config: dict,
) -> float:
    ...
```

配置：

```yaml
reward:
  type: final_score
  score_scale: 1000.0
  step_penalty: 0.0
  use_shaping: false
```

---

## 8. Gymnasium 单智能体训练环境

文件：

```text
mahjong_ai/env/gym_env.py
```

### 8.1 设计目的

第一版先实现单智能体环境：

```text
controlled_player = 0
其他 3 个玩家由 baseline agent 自动行动
```

这样可以直接使用 `MaskablePPO` 训练。

### 8.2 环境行为

`reset()`：

```text
1. rule_adapter.reset(seed) 创建新局；
2. 如果当前行动玩家不是 controlled_player，则调用 opponent agents 自动行动；
3. 直到轮到 controlled_player 或终局；
4. 构造 observation；
5. 构造 action mask；
6. 返回 obs, info。
```

`step(action)`：

```text
1. 检查 action 是否在 legal_actions 中；
2. 执行动作；
3. 如果未终局，则让其他玩家自动行动；
4. 直到再次轮到 controlled_player 或终局；
5. 计算 reward；
6. 构造 next observation；
7. 构造 next action mask；
8. 返回 obs, reward, terminated, truncated, info。
```

### 8.3 必须实现的类

```python
class MahjongSingleAgentEnv(gym.Env):
    def __init__(self, config: dict):
        ...

    def reset(self, seed=None, options=None):
        ...

    def step(self, action: int):
        ...

    def action_masks(self) -> np.ndarray:
        ...

    def render(self):
        ...
```

### 8.4 关键注意事项

```text
如果模型输出非法动作，训练阶段应 terminated=True 并给明显负奖励；
但正常情况下 MaskablePPO 不应输出非法动作；
每一步 info 中记录 legal_actions、action_mask、state_hash；
达到 max_steps_per_game 时 truncated=True；
终局时 terminated=True。
```

---

## 9. PettingZoo 多智能体环境

文件：

```text
mahjong_ai/env/pettingzoo_env.py
```

第一版可以只保留骨架，第二版再完整实现。

### 9.1 设计目的

多智能体 self-play 时使用。  
麻将是典型顺序行动游戏，适合 AEC 形式：

```text
agent_0 → agent_1 → agent_2 → agent_3 → 响应动作 → 下一个行动者
```

### 9.2 必须预留的接口

```python
class MahjongAECEnv(AECEnv):
    metadata = {"name": "custom_mahjong_v0"}

    def reset(self, seed=None, options=None):
        ...

    def observe(self, agent):
        ...

    def step(self, action):
        ...

    def render(self):
        ...

    def action_mask(self, agent) -> np.ndarray:
        ...
```

### 9.3 建议实现策略

```text
第一阶段：只实现 Gymnasium 单智能体环境；
第二阶段：复用 RuleAdapter、observation、actions、reward；
第三阶段：实现 AEC 多智能体版本；
第四阶段：接入 self-play 和历史模型池。
```

---

## 10. Baseline Agent

文件夹：

```text
mahjong_ai/agents/
```

### 10.1 必须实现的 Agent 接口

```python
class BaseAgent:
    def act(
        self,
        observation: np.ndarray,
        legal_actions: list[int],
        info: dict | None = None,
    ) -> int:
        raise NotImplementedError
```

### 10.2 RandomAgent

```text
从 legal_actions 中随机选择一个动作。
```

用于测试环境稳定性。

### 10.3 WinFirstAgent

策略：

```text
如果能胡，立即胡；
否则如果只能过，则过；
否则随机打牌。
```

### 10.4 HeuristicAgent

第一版启发式策略：

```text
1. 能胡则胡；
2. 能杠则根据配置选择是否杠；
3. 能碰则根据配置选择是否碰；
4. 出牌时优先打孤张；
5. 尽量保留对子和连续牌；
6. 如果无法判断，则随机合法出牌。
```

由于具体麻将规则非标准，启发式不必过强。  
它的主要作用是：

```text
验证环境；
提供 PPO 初始对手；
作为模型评估 baseline。
```

---

## 11. PPO 训练方案

文件：

```text
mahjong_ai/train/train_ppo.py
```

### 11.1 第一版算法

使用：

```text
MaskablePPO
MlpPolicy
单智能体环境
action mask
```

### 11.2 训练流程

```text
1. 读取 YAML 配置；
2. 设置随机种子；
3. 构造 RuleAdapter；
4. 构造 MahjongSingleAgentEnv；
5. 包装 VecEnv；
6. 初始化 MaskablePPO；
7. 定期评估；
8. 定期保存 checkpoint；
9. 训练结束保存 final_model。
```

### 11.3 配置示例

文件：

```text
configs/ppo_small.yaml
```

```yaml
seed: 2026

env:
  type: single_agent
  controlled_player: 0
  num_players: 4
  max_steps_per_game: 300
  obs_include_action_mask: false
  opponent_agent: heuristic
  rule_adapter: mock

reward:
  type: final_score
  score_scale: 1000.0
  step_penalty: 0.0
  use_shaping: false

action:
  n_tile_types: 34
  action_space_size: 128

model:
  policy: MlpPolicy
  net_arch: [512, 512, 256]

train:
  algorithm: MaskablePPO
  total_timesteps: 5000000
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 512
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5

logging:
  tensorboard_log: artifacts/logs/ppo_small
  checkpoint_dir: artifacts/checkpoints/ppo_small
  checkpoint_freq: 100000

eval:
  eval_freq: 100000
  num_games: 2000
  opponents:
    - random
    - win_first
    - heuristic
```

### 11.4 训练命令

```bash
python -m mahjong_ai.train.train_ppo --config configs/ppo_small.yaml
```

---

## 12. Self-play 方案

文件：

```text
mahjong_ai/train/self_play.py
```

### 12.1 不建议第一版直接做完全 self-play

原因：

```text
实现复杂；
多智能体训练不稳定；
容易策略坍塌；
难以判断模型是否真的变强。
```

建议路线：

```text
先训练 model_v1 对战 heuristic；
再让 model_v2 对战 heuristic + model_v1；
之后维护历史模型池。
```

### 12.2 历史模型池

设计：

```text
opponent_pool/
├── heuristic
├── model_v1
├── model_v2
└── model_v3
```

每次训练时随机采样对手：

```python
opponents = sample_opponents(pool, num_opponents=3)
```

采样策略：

```text
50% heuristic / rule-based bot；
30% 最近模型；
20% 历史模型。
```

### 12.3 升级条件

新模型进入模型池前必须满足：

```text
非法动作率为 0；
对 random 胜率明显高；
对 heuristic 平均收益不低；
对上一版模型不明显劣化；
平均局长无异常；
无明显拖局行为。
```

---

## 13. 评估系统

文件：

```text
mahjong_ai/eval/evaluate.py
mahjong_ai/eval/tournament.py
mahjong_ai/eval/metrics.py
```

### 13.1 评估指标

必须统计：

```text
avg_score：平均分；
win_rate：胜率或和牌率；
deal_in_rate：放铳率；
ready_rate：听牌率；
draw_rate：流局率；
avg_steps：平均局长；
illegal_action_count：非法动作次数；
seat_avg_score：不同座位平均分；
model_latency_ms：平均推理延迟。
```

### 13.2 评估命令

```bash
python -m mahjong_ai.eval.evaluate \
  --model artifacts/checkpoints/ppo_small/final_model.zip \
  --opponents heuristic,heuristic,heuristic \
  --num-games 20000 \
  --output artifacts/reports/ppo_small_eval.json
```

### 13.3 评估报告格式

```json
{
  "model": "ppo_small/final_model.zip",
  "num_games": 20000,
  "avg_score": 0.138,
  "win_rate": 0.274,
  "deal_in_rate": 0.196,
  "draw_rate": 0.112,
  "avg_steps": 73.4,
  "illegal_action_count": 0,
  "seat_avg_score": [0.12, 0.15, 0.13, 0.14],
  "model_latency_ms": {
    "mean": 3.2,
    "p95": 7.8
  }
}
```

---

## 14. 模型导出

文件夹：

```text
mahjong_ai/export/
```

### 14.1 导出目标

模型训练完成后需要导出到后端可加载格式：

```text
PyTorch checkpoint：训练继续使用；
TorchScript：Python / C++ 推理服务使用；
ONNX：轻量推理服务使用。
```

第一版建议：

```text
后端 AI 推理服务使用 Python + PyTorch；
直接加载 MaskablePPO 模型；
等接口稳定后再导出 ONNX。
```

### 14.2 导出命令

```bash
python -m mahjong_ai.export.export_onnx \
  --input artifacts/checkpoints/ppo_small/final_model.zip \
  --output artifacts/exported/mahjong_ppo_v1.onnx
```

```bash
python -m mahjong_ai.export.export_torchscript \
  --input artifacts/checkpoints/ppo_small/final_model.zip \
  --output artifacts/exported/mahjong_ppo_v1.pt
```

### 14.3 导出验证

必须实现：

```bash
python -m mahjong_ai.export.validate_export \
  --source artifacts/checkpoints/ppo_small/final_model.zip \
  --exported artifacts/exported/mahjong_ppo_v1.onnx \
  --num-samples 1000
```

验证内容：

```text
相同 observation 下输出动作基本一致；
输出动作必须在 legal_actions 内；
推理延迟满足线上要求；
导出模型文件存在且可加载。
```

---

## 15. 推理侧协议

训练侧需要提供 Predictor，供后端 AI 服务复用。

文件：

```text
mahjong_ai/inference/predictor.py
```

### 15.1 Predictor 接口

```python
class MahjongPredictor:
    def __init__(self, model_path: str, config: dict):
        ...

    def predict(
        self,
        observation: np.ndarray,
        legal_actions: list[int],
        deterministic: bool = True,
    ) -> dict:
        ...
```

返回：

```python
{
    "action": int,
    "confidence": float | None,
    "fallback_used": bool,
}
```

### 15.2 输入输出约束

输入：

```text
observation 必须是训练侧定义的固定长度向量；
legal_actions 必须由后端真实规则引擎生成；
predictor 内部必须再构造 action mask；
如果模型输出非法动作，必须 fallback。
```

输出：

```text
action 必须是 legal_actions 中的一个；
fallback_used 表示是否使用兜底策略；
confidence 可选，第一版可以返回 None。
```

---

## 16. 对局日志格式

训练侧应定义标准日志格式，方便后端按此保存线上牌谱。

### 16.1 单步日志

```json
{
  "game_id": "game_001",
  "step": 35,
  "player_id": 2,
  "state_hash_before": "abc",
  "observation": [0.0, 1.0],
  "legal_actions": [1, 5, 100],
  "action": 5,
  "action_source": "model",
  "model_version": "mahjong_ppo_v1",
  "state_hash_after": "def",
  "reward": 0.0
}
```

### 16.2 终局日志

```json
{
  "game_id": "game_001",
  "final_scores": [1.2, -0.5, -0.4, -0.3],
  "winner": 0,
  "draw": false,
  "total_steps": 83,
  "model_versions": {
    "0": "human",
    "1": "mahjong_ppo_v1",
    "2": "mahjong_ppo_v1",
    "3": "heuristic"
  }
}
```

这些日志后续可以用于：

```text
行为克隆；
离线评估；
失败局面回放；
模型异常诊断；
高水平玩家数据训练。
```

---

## 17. 测试要求

Codex 必须补充单元测试。  
至少包括：

### 17.1 actions 测试

```text
encode_action / decode_action 可逆；
build_action_mask 维度正确；
非法 action id 报错；
fallback_action 一定返回合法动作。
```

### 17.2 observation 测试

```text
observation 维度固定；
无 NaN / inf；
不包含其他玩家暗手牌；
不同 player_id 得到不同私有视角；
legal_action_mask 是否按配置拼入 observation。
```

### 17.3 env 测试

```text
reset 能返回合法 obs；
step 能执行合法动作；
action_masks 与 legal_actions 一致；
随机动作跑 1000 局不崩溃；
max_steps 后 truncated=True；
终局后 terminated=True。
```

### 17.4 predictor 测试

```text
predict 返回合法动作；
模型输出非法动作时 fallback；
空 legal_actions 时抛出明确异常；
推理输入维度错误时报错。
```

---

## 18. 开发顺序

Codex 应按以下顺序生成代码，避免一开始写复杂 self-play：

```text
Step 1：实现 actions.py；
Step 2：实现 RuleAdapter 协议和 MockRuleEngine；
Step 3：实现 observation.py；
Step 4：实现 RandomAgent / HeuristicAgent；
Step 5：实现 Gymnasium MahjongSingleAgentEnv；
Step 6：实现 check_env.py 和 run_random_games.py；
Step 7：实现 reward.py；
Step 8：实现 train_ppo.py；
Step 9：实现 evaluate.py / tournament.py；
Step 10：实现 predictor.py；
Step 11：实现 export_onnx.py / validate_export.py；
Step 12：预留 PettingZoo AEC 和 self_play.py。
```

第一版验收目标：

```text
1. pytest 全部通过；
2. 随机 bot 可跑 10000 局；
3. MaskablePPO debug 训练能启动；
4. 训练后模型可评估；
5. Predictor 可返回合法动作；
6. 模型文件可保存和加载。
```

---

## 19. 最小运行命令

### 19.1 安装依赖

```bash
pip install -r requirements.txt
```

### 19.2 运行测试

```bash
pytest -q
```

### 19.3 检查环境

```bash
python scripts/check_env.py
```

### 19.4 随机对局

```bash
python scripts/run_random_games.py --num-games 10000
```

### 19.5 Debug 训练

```bash
python -m mahjong_ai.train.train_ppo --config configs/ppo_debug.yaml
```

### 19.6 正式训练

```bash
python -m mahjong_ai.train.train_ppo --config configs/ppo_small.yaml
```

### 19.7 评估模型

```bash
python -m mahjong_ai.eval.evaluate \
  --model artifacts/checkpoints/ppo_small/final_model.zip \
  --opponents heuristic,heuristic,heuristic \
  --num-games 20000
```

### 19.8 导出模型

```bash
python -m mahjong_ai.export.export_onnx \
  --input artifacts/checkpoints/ppo_small/final_model.zip \
  --output artifacts/exported/mahjong_ppo_v1.onnx
```

---

## 20. 重要实现细节

### 20.1 关于非法动作

训练中出现非法动作一般说明：

```text
action mask 错；
legal_actions 编码错；
模型 predict 时没有传 action_masks；
环境 step 检查逻辑错。
```

处理方式：

```text
训练阶段：直接终止 episode，并记录错误；
推理阶段：fallback 到合法动作，并记录异常；
后端阶段：再次校验，绝不信任模型。
```

### 20.2 关于 observation 版本

训练侧必须维护 observation version：

```yaml
observation:
  version: obs_v1
```

一旦 observation 编码改变，必须：

```text
重新训练模型；
导出新模型版本；
后端传入 observation 时指定 version；
禁止旧模型加载新 observation。
```

### 20.3 关于模型版本

模型命名建议：

```text
mahjong_ppo_obs-v1_rule-v1_20260521
```

模型元数据保存：

```json
{
  "model_name": "mahjong_ppo_v1",
  "obs_version": "obs_v1",
  "rule_version": "rule_v1",
  "action_version": "action_v1",
  "train_steps": 5000000,
  "created_at": "2026-05-21"
}
```

### 20.4 关于真实规则接入

如果真实规则尚未实现，Codex 可以先写 `MockRuleEngine`。  
但所有接口都必须面向真实规则设计，不能把 mock 逻辑写死到环境里。

正确依赖方向：

```text
GymEnv → RuleAdapter → MockRuleEngine / RealRuleEngine
```

错误依赖方向：

```text
GymEnv 内部直接写具体麻将规则
```

---

## 21. 后续增强路线

第一版跑通后，再考虑以下增强：

```text
1. 引入 PettingZoo AEC 多智能体环境；
2. 引入历史模型池 self-play；
3. 引入行为克隆，使用真实玩家牌谱预训练；
4. 引入对手建模，估计其他玩家手牌分布；
5. 引入动作历史 Transformer；
6. 引入风险评估模块，降低放铳率；
7. 引入局面搜索，在关键回合做 rollout；
8. 引入模型蒸馏，降低推理延迟；
9. 引入 ONNX Runtime 部署；
10. 引入 A/B 测试和线上牌谱持续迭代。
```

---

## 22. Codex 实现要求总结

请 Codex 按以下要求实现训练侧代码：

```text
1. 使用 Python 3.10+；
2. 使用 Gymnasium + Stable-Baselines3-Contrib MaskablePPO；
3. 环境必须提供 action_masks()；
4. 所有规则调用都通过 RuleAdapter；
5. observation 必须固定维度；
6. 模型不能看到隐藏信息；
7. 实现 RandomAgent 和 HeuristicAgent；
8. 实现可运行的 MockRuleEngine；
9. 实现 PPO 训练脚本；
10. 实现评估脚本；
11. 实现 Predictor；
12. 实现导出与导出验证脚本；
13. 实现 pytest 测试；
14. 所有脚本都可以从项目根目录直接运行；
15. 训练产物统一保存到 artifacts/。
```

最终训练侧应支持完整闭环：

```text
pytest
→ random games
→ PPO debug train
→ PPO full train
→ evaluate
→ export
→ predictor inference
```

---

## 23. 技术依据与框架选择

本项目第一版选择 `Gymnasium + MaskablePPO`，是因为第一版可以把麻将训练问题转化为“单一受控玩家 + 其他玩家由 bot 自动行动”的离散动作强化学习问题，并通过 action mask 屏蔽非法动作。

第二版预留 `PettingZoo AEC`，是因为麻将属于顺序行动的多智能体不完全信息博弈，AEC 形式比简单并行动作接口更适合表达“当前轮到谁行动、谁可以响应、响应后如何跳转”的过程。

后期如果训练规模增大，可以将多智能体 self-play 迁移到 RLlib，以支持更复杂的多智能体训练、历史模型池和分布式采样。

---

## 24. 给 Codex 的一句话任务

请基于本文档，在当前 `training/` 文件夹中实现一个可运行的麻将 AI 训练系统：先用 MockRuleEngine 跑通 Gymnasium + MaskablePPO 训练闭环，再通过 RuleAdapter 预留真实规则引擎接入点，确保模型训练、评估、导出和推理接口均可独立运行。
