"""Tests for v12: river-sequence encoding (recent discards) + hand token sorting."""

from __future__ import annotations

import numpy as np

from mahjong_ai.env.gym_env import MahjongSingleAgentEnv
from mahjong_ai.env.observation import (
    RIVER_SEQ_DIM,
    SEAT_DIM,
    build_river_sequence,
    encode_hand_tokens,
    table_token_dim,
)
from mahjong_ai.env.actions import is_discard


def test_table_dim_switch():
    assert SEAT_DIM == 76
    assert RIVER_SEQ_DIM == 4 * 34 + 1
    assert table_token_dim({}) == 76
    assert table_token_dim({"observation": {"include_river_sequence": True}}) == 76 + RIVER_SEQ_DIM


def test_river_sequence_content():
    seq = build_river_sequence([1, 5, 9, 12, 17])
    # last 4 discards: 5, 9, 12, 17 (in order), 5th position = discard count
    assert seq[5] == 1.0 and seq[9 + 34] == 1.0 and seq[12 + 68] == 1.0 and seq[17 + 102] == 1.0
    assert abs(seq[136] - 5.0 / 40.0) < 1e-6
    # earlier discard (1) not in the window
    assert seq[1] == 0.0


def test_env_river_dims():
    cfg = {
        "observation": {
            "include_hand": True,
            "include_table": True,
            "include_value_features": True,
            "include_river_sequence": True,
        },
        "reward": {},
    }
    env = MahjongSingleAgentEnv(cfg)
    obs, info = env.reset(seed=3)
    assert obs["static"].shape[0] == 430
    assert obs["table"].shape == (4, 76 + RIVER_SEQ_DIM)
    legal = [a for a in info["legal_actions"] if is_discard(a)]
    if legal:
        tile = legal[0]
        obs2, _, _, _, _ = env.step(tile)
        assert obs2["table"].shape == obs["table"].shape


def test_env_without_river_keeps_76():
    cfg = {"observation": {"include_hand": True, "include_table": True}, "reward": {}}
    env = MahjongSingleAgentEnv(cfg)
    obs, _ = env.reset(seed=3)
    assert obs["table"].shape == (4, 76)


def test_hand_tokens_sorted():
    tokens, mask = encode_hand_tokens([9, 0, 5, 18])
    assert int(mask.sum()) == 4
    # token order is sorted ascending by tile
    active = [i for i in range(4) if mask[i]]
    tiles = [int(np.argmax(tokens[i])) for i in active]
    assert tiles == sorted(tiles)
    # wildcard flag survives
    wild_idx = next(i for i in active if tokens[i, 18] == 1.0)
    assert tokens[wild_idx, 34 + 1] == 1.0  # xiaoji flag
