from __future__ import annotations

from typing import Any

import numpy as np

from mahjong_ai.env.actions import ACTION_SPACE_SIZE, N_TILE_TYPES, build_action_mask
from mahjong_ai.rules.adapter import RuleAdapter

HISTORY_EVENT_TYPES = {
    "discard": 0,
    "chow": 1,
    "pong": 2,
    "kong_exposed": 3,
    "kong_concealed": 4,
    "kong_added": 5,
    "kong_draw": 6,
    "pass": 7,
    "win": 8,
}
HISTORY_EVENT_DIM = len(HISTORY_EVENT_TYPES) + 4 + 4 + (N_TILE_TYPES + 1) + 4
SEAT_DIM = N_TILE_TYPES + N_TILE_TYPES + 1 + 1 + 1 + 4 + 1


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


def is_history_observation(config: dict | None = None) -> bool:
    cfg = config or {}
    obs_cfg = cfg.get("observation", {})
    version = str(obs_cfg.get("version", cfg.get("obs_version", "")))
    return version in {"obs_v3_history", "v3_history"} or bool(obs_cfg.get("include_history", False))


def include_table_observation(config: dict | None = None) -> bool:
    cfg = config or {}
    obs_cfg = cfg.get("observation", {})
    return bool(obs_cfg.get("include_table", cfg.get("obs_include_table", False)))


def build_observation(
    rule_adapter: RuleAdapter,
    state: Any,
    player_id: int,
    config: dict | None = None,
) -> np.ndarray | dict[str, np.ndarray]:
    if is_history_observation(config):
        return build_history_observation(rule_adapter, state, player_id, config)
    static = build_static_observation(rule_adapter, state, player_id, config)
    if include_table_observation(config):
        return {
            "static": static.astype(np.float32),
            "table": build_table_tokens(rule_adapter, state, player_id),
        }
    return static


def build_table_tokens(rule_adapter: RuleAdapter, state: Any, player_id: int) -> np.ndarray:
    """Encode each seat as a token so an attention head can see the whole table.

    A seat token contains only public information: that seat's discards, open
    melds, score, dealer/current flags, relative position, and concealed hand
    count. The controlled player's private hand is not leaked here; it stays in
    the static observation.
    """

    public = rule_adapter.get_public_info(state)
    dealer = int(public["dealer"])
    current = int(public["current_player"])
    tokens = np.zeros((4, SEAT_DIM), dtype=np.float32)
    for seat in range(4):
        discards = _count_vec(list(public["discards"][seat]))
        meld_tiles = _count_vec([t for m in public["melds"][seat] for t in getattr(m, "tiles", [])])
        score = float(public["scores"][seat]) / 100.0
        relative = np.zeros(4, dtype=np.float32)
        relative[(seat - player_id) % 4] = 1.0
        hand_count = min(1.0, len(getattr(state, "hands", [[], [], [], []])[seat]) / 14.0)
        tokens[seat] = np.concatenate(
            [
                discards,
                meld_tiles,
                np.asarray([score], dtype=np.float32),
                np.asarray([1.0 if seat == dealer else 0.0], dtype=np.float32),
                np.asarray([1.0 if seat == current else 0.0], dtype=np.float32),
                relative,
                np.asarray([hand_count], dtype=np.float32),
            ]
        )
    return tokens.astype(np.float32)


def build_static_observation(
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


def build_history_observation(
    rule_adapter: RuleAdapter,
    state: Any,
    player_id: int,
    config: dict | None = None,
) -> dict[str, np.ndarray]:
    cfg = config or {}
    obs_cfg = cfg.get("observation", {})
    history_len = int(obs_cfg.get("history_len", cfg.get("history_len", 128)))
    static = build_static_observation(rule_adapter, state, player_id, config)
    history, mask = encode_public_history(getattr(state, "public_events", []), player_id, history_len)
    result: dict[str, np.ndarray] = {
        "static": static.astype(np.float32),
        "history": history.astype(np.float32),
        "history_mask": mask.astype(np.float32),
    }
    if include_table_observation(config):
        result["table"] = build_table_tokens(rule_adapter, state, player_id)
    return result


def encode_public_history(
    events: list[dict],
    player_id: int,
    history_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    history = np.zeros((history_len, HISTORY_EVENT_DIM), dtype=np.float32)
    mask = np.zeros((history_len,), dtype=np.float32)
    selected = list(events[-history_len:])
    offset = history_len - len(selected)
    for i, event in enumerate(selected):
        row = history[offset + i]
        event_type = str(event.get("type", ""))
        if event_type in HISTORY_EVENT_TYPES:
            row[HISTORY_EVENT_TYPES[event_type]] = 1.0
        cursor = len(HISTORY_EVENT_TYPES)
        actor = int(event.get("player", 0))
        if 0 <= actor < 4:
            row[cursor + actor] = 1.0
        cursor += 4
        row[cursor + ((actor - player_id) % 4)] = 1.0
        cursor += 4
        tile = event.get("tile")
        if tile is None:
            row[cursor + N_TILE_TYPES] = 1.0
        else:
            tile_int = int(tile)
            if 0 <= tile_int < N_TILE_TYPES:
                row[cursor + tile_int] = 1.0
        cursor += N_TILE_TYPES + 1
        row[cursor] = min(1.0, float(event.get("step", 0)) / 300.0)
        row[cursor + 1] = min(1.0, float(event.get("wall", 0)) / 136.0)
        target = event.get("target")
        row[cursor + 2] = 0.0 if target is None else ((int(target) - player_id) % 4) / 3.0
        row[cursor + 3] = 1.0 if bool(event.get("xiaoji_disabled", False)) else 0.0
        mask[offset + i] = 1.0
    return history, mask


def validate_observation(obs: np.ndarray, expected_dim: int) -> None:
    if obs.shape != (expected_dim,):
        raise ValueError(f"observation shape {obs.shape} != ({expected_dim},)")
    if not np.isfinite(obs).all():
        raise ValueError("observation contains NaN or inf")
