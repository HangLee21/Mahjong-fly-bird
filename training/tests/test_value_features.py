"""Tests for v10 value features: remaining-tile counts + hand quality scalars."""

from __future__ import annotations

from mahjong_ai.env.gym_env import MahjongSingleAgentEnv
from mahjong_ai.env.observation import VALUE_FEATURE_DIM, get_observation_dim


def _env(cfg_extra: dict):
    cfg = {"reward": {}, **cfg_extra}
    return MahjongSingleAgentEnv(cfg)


def test_value_features_dim():
    cfg = {"observation": {"include_hand": True, "include_table": True, "include_value_features": True}}
    env = _env(cfg)
    obs, _ = env.reset(seed=3)
    assert obs["static"].shape[0] == get_observation_dim(cfg) == 394 + VALUE_FEATURE_DIM
    assert VALUE_FEATURE_DIM == 34 + 2


def test_value_features_ranges():
    cfg = {"observation": {"include_hand": True, "include_table": True, "include_value_features": True}}
    env = _env(cfg)
    obs, _ = env.reset(seed=3)
    vf = obs["static"][-VALUE_FEATURE_DIM:]
    remaining, scalars = vf[:34], vf[34:]
    assert remaining.min() >= 0.0 and remaining.max() <= 1.0
    assert scalars.min() >= 0.0 and scalars.max() <= 1.0
    # a fresh hand + empty table: nearly every tile still available
    assert float(remaining.sum()) > 30.0 / 4.0


def test_value_features_off_keeps_394():
    cfg = {"observation": {"include_hand": True, "include_table": True}}
    env = _env(cfg)
    obs, _ = env.reset(seed=3)
    assert obs["static"].shape[0] == 394


def test_remaining_decreases_after_discard():
    cfg = {"observation": {"include_hand": True, "include_table": True, "include_value_features": True}}
    env = _env(cfg)
    obs, info = env.reset(seed=7)
    import numpy as np

    from mahjong_ai.env.actions import is_discard

    legal = list(info["legal_actions"])
    discards = [a for a in legal if is_discard(a)]
    assert discards, "expected at least one discard action"
    tile = int(discards[0])
    before = obs["static"][-VALUE_FEATURE_DIM:][:34][tile]
    obs2, _, terminated, truncated, _ = env.step(tile)
    after = obs2["static"][-VALUE_FEATURE_DIM:][:34][tile]
    # discarding a tile makes it seen -> remaining for that tile drops
    assert after <= before + 1e-6
