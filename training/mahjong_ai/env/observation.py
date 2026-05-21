from __future__ import annotations

from typing import Any

import numpy as np

from mahjong_ai.env.actions import ACTION_SPACE_SIZE, N_TILE_TYPES, build_action_mask
from mahjong_ai.rules.adapter import RuleAdapter


def _count_vec(tiles: list[int], denom: float = 4.0) -> np.ndarray:
    vec = np.zeros(N_TILE_TYPES, dtype=np.float32)
    for tile in tiles:
        vec[tile] += 1.0
    return vec / denom


def get_observation_dim(config: dict | None = None) -> int:
    cfg = config or {}
    include_mask = cfg.get("obs_include_action_mask", False)
    base = (
        N_TILE_TYPES
        + N_TILE_TYPES
        + 4 * N_TILE_TYPES
        + 4 * N_TILE_TYPES
        + 4
        + 4
        + 4
        + 4
        + (N_TILE_TYPES + 1)
        + 3
    )
    return base + (ACTION_SPACE_SIZE if include_mask else 0)


def build_observation(
    rule_adapter: RuleAdapter,
    state: Any,
    player_id: int,
    config: dict | None = None,
) -> np.ndarray:
    cfg = config or {}
    public = rule_adapter.get_public_info(state)
    private = rule_adapter.get_private_info(state, player_id)
    legal_actions = rule_adapter.get_legal_actions(state, player_id)

    hand_counts = _count_vec(private["hand"])
    self_meld_counts = _count_vec([t for m in public["melds"][player_id] for t in m.tiles])
    discard_counts = np.concatenate([_count_vec(d) for d in public["discards"]]).astype(np.float32)
    open_meld_counts = np.concatenate(
        [_count_vec([t for m in melds for t in m.tiles]) for melds in public["melds"]]
    ).astype(np.float32)
    scores = np.asarray(public["scores"], dtype=np.float32) / 100.0
    dealer = np.zeros(4, dtype=np.float32)
    dealer[public["dealer"]] = 1.0
    current = np.zeros(4, dtype=np.float32)
    current[public["current_player"]] = 1.0
    relative = np.zeros(4, dtype=np.float32)
    relative[(public["current_player"] - player_id) % 4] = 1.0
    last = np.zeros(N_TILE_TYPES + 1, dtype=np.float32)
    if public["last_discard"] is None:
        last[-1] = 1.0
    else:
        last[public["last_discard"]] = 1.0
    round_info = np.asarray(
        [
            public["remaining_wall"] / 136.0,
            len(public["kong_pool"]) / 2.0,
            1.0 if public["xiaoji_disabled"] else 0.0,
        ],
        dtype=np.float32,
    )

    parts = [
        hand_counts,
        self_meld_counts,
        discard_counts,
        open_meld_counts,
        scores,
        dealer,
        current,
        relative,
        last,
        round_info,
    ]
    if cfg.get("obs_include_action_mask", False):
        parts.append(build_action_mask(legal_actions).astype(np.float32))
    obs = np.concatenate(parts).astype(np.float32)
    validate_observation(obs, get_observation_dim(cfg))
    return obs


def validate_observation(obs: np.ndarray, expected_dim: int) -> None:
    if obs.shape != (expected_dim,):
        raise ValueError(f"observation shape {obs.shape} != ({expected_dim},)")
    if not np.isfinite(obs).all():
        raise ValueError("observation contains NaN or inf")

