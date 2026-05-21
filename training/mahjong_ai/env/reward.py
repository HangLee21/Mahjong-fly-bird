from __future__ import annotations

from typing import Any

from mahjong_ai.env.actions import (
    ACTION_KONG_ADDED,
    ACTION_KONG_CONCEALED,
    ACTION_KONG_EXPOSED,
    is_discard,
)
from mahjong_ai.rules.adapter import RuleAdapter
from mahjong_ai.rules.flybird import WILDCARD
from mahjong_ai.rules.shanten import best_shanten


def compute_reward(
    prev_state: Any,
    next_state: Any,
    player_id: int,
    rule_adapter: RuleAdapter,
    config: dict | None = None,
    action: int | None = None,
) -> float:
    cfg = config or {}
    scale = float(cfg.get("score_scale", 1.0))
    step_penalty = float(cfg.get("step_penalty", 0.0))
    prev = rule_adapter.get_scores(prev_state)[player_id]
    nxt = rule_adapter.get_scores(next_state)[player_id]
    reward = (nxt - prev) / scale
    reward += _action_shaping(prev_state, action, cfg)
    reward += _shanten_shaping(prev_state, next_state, player_id, cfg)
    if not rule_adapter.is_terminal(next_state):
        reward -= step_penalty
    return float(reward)


def _action_shaping(prev_state: Any, action: int | None, cfg: dict) -> float:
    if action is None:
        return 0.0
    reward = 0.0
    if is_discard(action) and action == WILDCARD:
        if getattr(prev_state, "xiaoji_disabled", False):
            reward -= float(cfg.get("discard_dead_xiaoji_penalty", 0.0))
        else:
            reward -= float(cfg.get("discard_live_xiaoji_penalty", 0.0))
    if action == ACTION_KONG_CONCEALED:
        reward += float(cfg.get("concealed_kong_bonus", 0.0))
    elif action == ACTION_KONG_EXPOSED:
        reward -= float(cfg.get("exposed_kong_penalty", 0.0))
    elif action == ACTION_KONG_ADDED:
        reward -= float(cfg.get("added_kong_penalty", 0.0))
    return reward


def _shanten_shaping(prev_state: Any, next_state: Any, player_id: int, cfg: dict) -> float:
    improvement_bonus = float(cfg.get("shanten_improvement_bonus", 0.0))
    regression_penalty = float(cfg.get("shanten_regression_penalty", 0.0))
    ready_bonus = float(cfg.get("ready_bonus", 0.0))
    if improvement_bonus == 0.0 and regression_penalty == 0.0 and ready_bonus == 0.0:
        return 0.0
    prev_open = len(getattr(prev_state, "melds", [[] for _ in range(4)])[player_id])
    next_open = len(getattr(next_state, "melds", [[] for _ in range(4)])[player_id])
    prev_wild = not bool(getattr(prev_state, "xiaoji_disabled", False))
    next_wild = not bool(getattr(next_state, "xiaoji_disabled", False))
    prev_shanten = best_shanten(prev_state.hands[player_id], open_melds=prev_open, wildcard_enabled=prev_wild)
    next_shanten = best_shanten(next_state.hands[player_id], open_melds=next_open, wildcard_enabled=next_wild)
    delta = prev_shanten - next_shanten
    reward = 0.0
    if delta > 0:
        reward += improvement_bonus * delta
    elif delta < 0:
        reward -= regression_penalty * abs(delta)
    if prev_shanten > 0 and next_shanten == 0:
        reward += ready_bonus
    return reward
