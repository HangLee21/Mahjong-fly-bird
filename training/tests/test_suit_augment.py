"""Tests for v11 suit augmentation (万筒互换, wildcard-safe)."""

from __future__ import annotations

from mahjong_ai.env.gym_env import _flip_state, _flip_tile, MahjongSingleAgentEnv
from mahjong_ai.rules.flybird import GameState


def test_flip_map_rules():
    assert _flip_tile(0) == 9 and _flip_tile(9) == 0
    assert _flip_tile(5) == 14 and _flip_tile(14) == 5
    assert _flip_tile(18) == 18  # 癞子（一条）不动
    assert _flip_tile(27) == 27  # 字牌不动
    assert _flip_tile(20) == 20  # 条子（非癞子）不动


def test_flip_is_involution():
    for tile in range(34):
        assert _flip_tile(_flip_tile(tile)) == tile


def test_flip_state_keeps_wildcard_and_maps_suits():
    state = GameState(
        hands=[[0, 18, 20]],
        wall=[9, 27],
        kong_pool=[5],
        discards=[[3], [], [], []],
        melds=[[], [], [], []],
        scores=[0, 0, 0, 0],
    )
    _flip_state(state, True)
    assert 18 in state.hands[0]  # 癞子保持
    assert 9 in state.hands[0]   # 0 -> 9
    assert 20 in state.hands[0]  # 条子不动
    assert state.wall == [0, 27]  # 9 -> 0, 27 不动
    assert state.kong_pool == [14]  # 5 -> 14
    assert state.discards[0] == [12]  # 3 -> 12


def test_env_augment_steps_and_dims():
    cfg = {
        "observation": {"include_hand": True, "include_table": True, "include_value_features": True},
        "reward": {},
        "augment_suit": True,
    }
    env = MahjongSingleAgentEnv(cfg)
    obs, info = env.reset(seed=1)
    assert obs["static"].shape[0] == 430
    legal = list(info["legal_actions"])
    assert legal
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(legal[0])
        legal = list(info["legal_actions"])
        if terminated or truncated:
            obs, info = env.reset(seed=2)
            legal = list(info["legal_actions"])
        assert legal


def test_env_without_augment_unchanged():
    cfg = {"observation": {"include_hand": True, "include_table": True}, "reward": {}}
    env = MahjongSingleAgentEnv(cfg)
    obs, _ = env.reset(seed=1)
    assert obs["static"].shape[0] == 394
