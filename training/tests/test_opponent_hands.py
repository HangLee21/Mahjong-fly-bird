"""Tests for v15 perfect-information approximation (Suphx-style opponent hands)."""

from __future__ import annotations

import numpy as np

from mahjong_ai.env.gym_env import MahjongSingleAgentEnv
from mahjong_ai.env.observation import OPP_HANDS_DIM, build_opponent_hands, get_observation_dim
from mahjong_ai.rules.flybird import FlybirdRuleEngine


def test_opp_hands_dim():
    assert OPP_HANDS_DIM == 3 * 34
    cfg = {
        "observation": {
            "include_hand": True,
            "include_table": True,
            "include_value_features": True,
            "include_river_sequence": True,
            "include_opponent_hands": True,
        },
        "reward": {},
    }
    env = MahjongSingleAgentEnv(cfg)
    obs, _ = env.reset(seed=1)
    assert obs["static"].shape[0] == get_observation_dim(cfg) == 430 + OPP_HANDS_DIM


def test_visible_fills_values_closed_zeros():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=3)
    state.hands[1] = [0, 1, 2]
    state.hands[2] = [18, 18]
    state.hands[3] = [27]
    opened = build_opponent_hands(state, 0, visible=True)
    closed = build_opponent_hands(state, 0, visible=False)
    assert float(opened.sum()) > 0.0
    assert float(closed.sum()) == 0.0
    # seat 1 (slot 0) has tiles 0,1,2 -> counts 1 each
    assert abs(opened[0] - 1 / 4) < 1e-6
    assert abs(opened[1] - 1 / 4) < 1e-6
    # seat 2 (slot 1) has 18 twice
    assert abs(opened[34 + 18] - 2 / 4) < 1e-6
    # seat 3 (slot 2) has 27
    assert abs(opened[68 + 27] - 1 / 4) < 1e-6


def test_env_steps_with_opponent_hands():
    cfg = {
        "observation": {
            "include_hand": True,
            "include_table": True,
            "include_value_features": True,
            "include_opponent_hands": True,
        },
        "reward": {},
    }
    env = MahjongSingleAgentEnv(cfg)
    obs, info = env.reset(seed=5)
    assert obs["static"].shape[0] == 394 + 36 + OPP_HANDS_DIM
    legal = list(info["legal_actions"])
    assert legal
    for _ in range(6):
        obs, _, terminated, truncated, info = env.step(legal[0])
        legal = list(info["legal_actions"])
        if terminated or truncated:
            obs, info = env.reset(seed=6)
            legal = list(info["legal_actions"])
        assert legal
