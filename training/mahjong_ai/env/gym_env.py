from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    gym = None
    spaces = None

from mahjong_ai.agents.heuristic_agent import HeuristicAgent, WinFirstAgent
from mahjong_ai.agents.opponent_pool import OpponentPool
from mahjong_ai.agents.random_agent import RandomAgent
from mahjong_ai.env.actions import ACTION_PASS, ACTION_SPACE_SIZE, build_action_mask
from mahjong_ai.env.observation import (
    HAND_MAX_TILES,
    HAND_TOKEN_DIM,
    HISTORY_EVENT_DIM,
    SEAT_DIM,
    build_observation,
    get_observation_dim,
    include_hand_observation,
    include_table_observation,
    is_history_observation,
    table_token_dim,
)
from mahjong_ai.env.reward import compute_reward
from mahjong_ai.rules.flybird import FlybirdRuleEngine

# 万筒互换映射：t in [0,8] <-> t+9 in [9,17]。条子(18-26)与字牌(27-33)不动，
# 癞子(18, 一条)语义保持。这是麻将牌效的等价变换（suit 不变性），用于训练时
# 数据增强（augment_suit）。整个 GameState 翻转后，obs/action/reward 全部自洽。
_SUIT_FLIP_MAP = list(range(34))
for _i in range(9):
    _SUIT_FLIP_MAP[_i] = _i + 9
    _SUIT_FLIP_MAP[_i + 9] = _i


def _flip_tile(tile: int) -> int:
    tile = int(tile)
    return _SUIT_FLIP_MAP[tile] if 0 <= tile < 34 else tile


def _flip_state(state: Any, flip: bool) -> Any:
    """In-place suit flip of every tile-carrying field of a GameState."""
    if not flip:
        return state
    state.hands = [[_flip_tile(t) for t in hand] for hand in state.hands]
    state.discards = [[_flip_tile(t) for t in discards] for discards in state.discards]
    state.wall = [_flip_tile(t) for t in state.wall]
    state.kong_pool = [_flip_tile(t) for t in state.kong_pool]
    for melds in state.melds:
        for meld in melds:
            meld.tiles = [_flip_tile(t) for t in meld.tiles]
            if meld.wildcard_as is not None:
                meld.wildcard_as = _flip_tile(meld.wildcard_as)
    if state.last_discard is not None:
        state.last_discard = _flip_tile(state.last_discard)
    if state.last_kong_tile is not None:
        state.last_kong_tile = _flip_tile(state.last_kong_tile)
    if state.pending is not None:
        state.pending.tile = _flip_tile(state.pending.tile)
    return state


class MahjongSingleAgentEnv(gym.Env if gym else object):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config: dict | None = None):
        if gym is None:
            raise ImportError("gymnasium is required for MahjongSingleAgentEnv")
        self.config = config or {}
        self.controlled_player = int(self.config.get("controlled_player", 0))
        self.augment_suit = bool(self.config.get("augment_suit", False))
        self.max_steps = int(self.config.get("max_steps_per_game", 300))
        self.rule_adapter = self.config.get("rule_adapter") or FlybirdRuleEngine(
            allow_chow=bool(self.config.get("allow_chow", True))
        )
        self.opponent_kind = str(self.config.get("opponent_agent", "heuristic"))
        self.opponent_pool = self._make_opponent_pool()
        self.opponents = self._make_opponents(self.opponent_kind)
        self.state: Any | None = None
        self._last_obs: Any | None = None
        include_table = include_table_observation(self.config)
        include_hand = include_hand_observation(self.config)
        if is_history_observation(self.config) or include_table or include_hand:
            history_len = int(self.config.get("observation", {}).get("history_len", self.config.get("history_len", 128)))
            spaces_dict = {
                "static": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(get_observation_dim(self.config),),
                    dtype=np.float32,
                )
            }
            if include_table:
                spaces_dict["table"] = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(4, table_token_dim(self.config)),
                    dtype=np.float32,
                )
            if include_hand:
                spaces_dict["hand"] = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(HAND_MAX_TILES, HAND_TOKEN_DIM),
                    dtype=np.float32,
                )
                spaces_dict["hand_mask"] = spaces.Box(low=0.0, high=1.0, shape=(HAND_MAX_TILES,), dtype=np.float32)
            if is_history_observation(self.config):
                spaces_dict["history"] = spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(history_len, HISTORY_EVENT_DIM),
                    dtype=np.float32,
                )
                spaces_dict["history_mask"] = spaces.Box(low=0.0, high=1.0, shape=(history_len,), dtype=np.float32)
            self.observation_space = spaces.Dict(spaces_dict)
        else:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(get_observation_dim(self.config),),
                dtype=np.float32,
            )
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.opponent_kind == "pool":
            self.opponents = self._make_opponents(self.opponent_kind)
        self.state = self.rule_adapter.reset(seed)
        if self.augment_suit:
            flip = (seed if seed is not None else 0) % 2 == 1
            _flip_state(self.state, flip)
        self._auto_play_until_controlled()
        self._auto_pass_controlled_forced()
        obs = self._obs()
        return obs, self._info()

    def step(self, action: int):
        assert self.state is not None
        legal = self.rule_adapter.get_legal_actions(self.state, self.controlled_player)
        prev_state = self.rule_adapter.clone_state(self.state)
        if int(action) not in legal:
            obs = self._obs()
            return obs, -10.0, True, False, {**self._info(), "illegal_action": int(action)}
        self.state = self.rule_adapter.step(self.state, self.controlled_player, int(action))
        self._auto_play_until_controlled()
        self._auto_pass_controlled_forced()
        reward = compute_reward(
            prev_state,
            self.state,
            self.controlled_player,
            self.rule_adapter,
            self.config.get("reward", {}),
            action=int(action),
        )
        terminated = self.rule_adapter.is_terminal(self.state)
        truncated = bool(not terminated and self.state.step_count >= self.max_steps)
        if truncated:
            self.state.terminal = True
            self.state.draw = True
        obs = self._obs()
        return obs, reward, terminated, truncated, self._info()

    def action_masks(self) -> np.ndarray:
        assert self.state is not None
        return build_action_mask(self.rule_adapter.get_legal_actions(self.state, self.controlled_player))

    def render(self):
        assert self.state is not None
        return (
            f"player={self.rule_adapter.get_current_player(self.state)} "
            f"wall={len(self.state.wall)} scores={self.state.scores}"
        )

    def _make_opponents(self, kind: str):
        if kind == "pool":
            assert self.opponent_pool is not None
            return self.opponent_pool.sample_table(self.controlled_player)
        agents = []
        for seat in range(4):
            if seat == self.controlled_player:
                agents.append(None)
            elif kind == "random":
                agents.append(RandomAgent(seed=seat))
            elif kind == "win_first":
                agents.append(WinFirstAgent(seed=seat))
            else:
                agents.append(HeuristicAgent(seed=seat))
        return agents

    def _make_opponent_pool(self) -> OpponentPool | None:
        if self.opponent_kind != "pool":
            return None
        pool_cfg = self.config.get("opponent_pool", {})
        seed = self.config.get("seed_offset", self.config.get("seed"))
        return OpponentPool(pool_cfg, seed=int(seed) if seed is not None else None)

    def _auto_play_until_controlled(self) -> None:
        assert self.state is not None
        guard = 0
        while (
            not self.rule_adapter.is_terminal(self.state)
            and self.rule_adapter.get_current_player(self.state) != self.controlled_player
            and self.state.step_count < self.max_steps
        ):
            player = self.rule_adapter.get_current_player(self.state)
            legal = self.rule_adapter.get_legal_actions(self.state, player)
            opponent = self.opponents[player]
            # Rule-based opponents only read `info`; skip the (expensive)
            # observation build unless the opponent actually needs it.
            if getattr(opponent, "uses_observation", False):
                obs = build_observation(self.rule_adapter, self.state, player, self.config)
            else:
                obs = None
            info = {**self._info_for(player), "hand": self.state.hands[player]}
            action = opponent.act(obs, legal, info)
            self.state = self.rule_adapter.step(self.state, player, action)
            guard += 1
            if guard > self.max_steps * 4:
                self.state.terminal = True
                self.state.draw = True
                break

    def _auto_pass_controlled_forced(self) -> None:
        assert self.state is not None
        guard = 0
        while (
            not self.rule_adapter.is_terminal(self.state)
            and self.rule_adapter.get_current_player(self.state) == self.controlled_player
            and self.rule_adapter.get_legal_actions(self.state, self.controlled_player) == [ACTION_PASS]
            and self.state.step_count < self.max_steps
        ):
            self.state = self.rule_adapter.step(self.state, self.controlled_player, ACTION_PASS)
            self._auto_play_until_controlled()
            guard += 1
            if guard > self.max_steps:
                self.state.terminal = True
                self.state.draw = True
                break

    def _obs(self) -> Any:
        assert self.state is not None
        self._last_obs = build_observation(self.rule_adapter, self.state, self.controlled_player, self.config)
        return self._last_obs

    def _info(self) -> dict:
        return self._info_for(self.controlled_player)

    def _info_for(self, player_id: int) -> dict:
        assert self.state is not None
        legal = self.rule_adapter.get_legal_actions(self.state, player_id)
        return {
            "legal_actions": legal,
            "action_mask": build_action_mask(legal),
            "state_hash": self.rule_adapter.get_state_hash(self.state),
            "scores": self.rule_adapter.get_scores(self.state),
            "winner": self.rule_adapter.get_winner(self.state),
            "winners": list(getattr(self.state, "winners", [])),
            "draw": bool(getattr(self.state, "draw", False)),
            "step_count": int(getattr(self.state, "step_count", 0)),
            "win_type": getattr(self.state, "win_type", None),
            "payer": getattr(self.state, "payer", None),
            "win_points": getattr(self.state, "win_points", 0.0),
            "win_names": getattr(self.state, "win_names", []),
            "hand": list(getattr(self.state, "hands", [[] for _ in range(4)])[player_id]),
            "open_melds": len(getattr(self.state, "melds", [[] for _ in range(4)])[player_id]),
            "xiaoji_disabled": bool(getattr(self.state, "xiaoji_disabled", False)),
            "last_discard": getattr(self.state, "last_discard", None),
        }
