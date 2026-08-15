from __future__ import annotations

from typing import Any

from mahjong_ai.env.actions import (
    ACTION_KONG_ADDED,
    ACTION_KONG_CONCEALED,
    ACTION_KONG_EXPOSED,
    is_discard,
)
from mahjong_ai.rules.adapter import RuleAdapter
from mahjong_ai.rules.flybird import HONORS, WILDCARD, counts, score_hand, tile_suit
from mahjong_ai.rules.shanten import best_shanten, fast_hand_value


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
    reward += _action_shaping(prev_state, player_id, action, cfg)
    reward += _shanten_shaping(prev_state, next_state, player_id, cfg)
    reward += _hand_goal_shaping(prev_state, next_state, player_id, cfg)
    reward += _terminal_win_shaping(next_state, player_id, cfg)
    if not rule_adapter.is_terminal(next_state):
        reward -= step_penalty
    return float(reward)


def discard_danger_score(state: Any, tile: int, player_id: int) -> float:
    """Estimate how dangerous discarding `tile` is, from public information.

    Higher means more dangerous (opponents are more likely to be waiting on it).
    A tile is fully safe when all four copies are visible in public discards or
    open melds. Middle-number suited tiles are the most dangerous, honors and
    terminals less so. The player's own hand is intentionally not considered
    here because we are scoring the tile about to be discarded.
    """

    visible = 0
    for seat, discards in enumerate(getattr(state, "discards", [[] for _ in range(4)])):
        visible += list(discards).count(int(tile))
    for melds in getattr(state, "melds", [[] for _ in range(4)]):
        for meld in melds:
            visible += list(getattr(meld, "tiles", [])).count(int(tile))

    live = max(0, 4 - visible)
    if live == 0:
        return 0.0
    if int(tile) >= 27:
        base = 0.55
    elif int(tile) % 9 + 1 in (1, 9):
        base = 0.4
    elif int(tile) % 9 + 1 in (2, 8):
        base = 0.7
    else:
        base = 1.0
    return base * (live / 4.0)


def _action_shaping(prev_state: Any, player_id: int, action: int | None, cfg: dict) -> float:
    if action is None:
        return 0.0
    reward = 0.0
    if is_discard(action) and action == WILDCARD:
        if getattr(prev_state, "xiaoji_disabled", False):
            reward -= float(cfg.get("discard_dead_xiaoji_penalty", 0.0))
        else:
            reward -= float(cfg.get("discard_live_xiaoji_penalty", 0.0))
    if is_discard(action):
        hand = list(getattr(prev_state, "hands", [[] for _ in range(4)])[player_id])
        wildcard_enabled = not bool(getattr(prev_state, "xiaoji_disabled", False))
        reward += discard_preference_reward(action, hand, wildcard_enabled, cfg)
        danger = discard_danger_score(prev_state, action, player_id)
        reward -= float(cfg.get("discard_danger_penalty", 0.0)) * danger
    if action == ACTION_KONG_CONCEALED:
        reward += float(cfg.get("concealed_kong_bonus", 0.0))
    elif action == ACTION_KONG_EXPOSED:
        reward -= float(cfg.get("exposed_kong_penalty", 0.0))
    elif action == ACTION_KONG_ADDED:
        if bool(cfg.get("added_kong_as_concealed", False)):
            reward += float(cfg.get("added_kong_as_concealed_bonus", cfg.get("concealed_kong_bonus", 0.0)))
            hand = list(getattr(prev_state, "hands", [[] for _ in range(4)])[player_id])
            open_melds = len(getattr(prev_state, "melds", [[] for _ in range(4)])[player_id])
            wildcard_enabled = not bool(getattr(prev_state, "xiaoji_disabled", False))
            if best_shanten(hand, open_melds=open_melds, wildcard_enabled=wildcard_enabled) <= 1:
                reward += float(cfg.get("added_kong_ready_bonus", 0.0))
        else:
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


def _terminal_win_shaping(next_state: Any, player_id: int, cfg: dict) -> float:
    if not bool(getattr(next_state, "terminal", False)):
        return 0.0
    winners = list(getattr(next_state, "winners", []))
    if player_id not in winners:
        return 0.0

    reward = 0.0
    reward += float(cfg.get("terminal_win_bonus", 0.0))
    if getattr(next_state, "win_type", None) == "self_draw":
        reward += float(cfg.get("self_draw_bonus", 0.0))
    else:
        reward += float(cfg.get("ron_bonus", 0.0))

    fan_bonus = float(cfg.get("fan_bonus", 0.0))
    point_bonus = float(cfg.get("point_bonus", 0.0))
    if fan_bonus or point_bonus:
        score = score_hand(
            next_state,
            player_id,
            getattr(next_state, "last_discard", None),
            getattr(next_state, "win_type", None) == "self_draw",
        )
        reward += fan_bonus * float(score.get("fan", 0))
        reward += point_bonus * float(score.get("points", 0.0))
    return reward


def _hand_goal_shaping(prev_state: Any, next_state: Any, player_id: int, cfg: dict) -> float:
    improvement_bonus = float(cfg.get("hand_goal_improvement_bonus", 0.0))
    regression_penalty = float(cfg.get("hand_goal_regression_penalty", 0.0))
    if improvement_bonus == 0.0 and regression_penalty == 0.0:
        return 0.0

    prev_scores = _hand_goal_scores(prev_state, player_id)
    next_scores = _hand_goal_scores(next_state, player_id)
    mode = str(cfg.get("hand_goal_mode", "committed"))
    if mode == "best_delta":
        delta = max(next - prev for prev, next in zip(prev_scores, next_scores))
        target = max(range(len(prev_scores)), key=lambda i: prev_scores[i])
    else:
        target = max(range(len(prev_scores)), key=lambda i: prev_scores[i])
        delta = next_scores[target] - prev_scores[target]
    if delta > 0:
        reward = improvement_bonus * delta
    if delta < 0:
        reward = regression_penalty * delta
    if delta == 0:
        reward = 0.0
    switch_penalty = float(cfg.get("hand_goal_switch_penalty", 0.0))
    if switch_penalty > 0.0:
        next_target = max(range(len(next_scores)), key=lambda i: next_scores[i])
        if next_target != target:
            reward -= switch_penalty
    return reward


def _best_hand_goal_score(state: Any, player_id: int) -> float:
    return max(_hand_goal_scores(state, player_id))


def _hand_goal_scores(state: Any, player_id: int) -> list[float]:
    hand = list(getattr(state, "hands", [[] for _ in range(4)])[player_id])
    melds = list(getattr(state, "melds", [[] for _ in range(4)])[player_id])
    extra_tiles = [tile for meld in melds for tile in getattr(meld, "tiles", [])]
    return hand_goal_scores_for_tiles(
        hand,
        extra_tiles=extra_tiles,
        open_melds=len(melds),
        xiaoji_disabled=bool(getattr(state, "xiaoji_disabled", False)),
    )


def hand_goal_scores_for_tiles(
    hand: list[int],
    *,
    extra_tiles: list[int] | None = None,
    open_melds: int = 0,
    xiaoji_disabled: bool = False,
) -> list[float]:
    wildcard_enabled = not xiaoji_disabled
    all_tiles = list(hand) + list(extra_tiles or [])
    if not all_tiles:
        return [0.0]

    return [
        _standard_goal_score(hand, open_melds, wildcard_enabled),
        _seven_pairs_goal_score(hand, wildcard_enabled) if open_melds == 0 else -99.0,
        _flush_goal_score(all_tiles),
        _triplet_goal_score(all_tiles),
    ]


def discard_preference_reward(tile: int, hand: list[int], wildcard_enabled: bool, cfg: dict) -> float:
    if wildcard_enabled and tile == WILDCARD:
        return 0.0
    score = discard_preference_score(tile, hand, wildcard_enabled)
    return score * float(cfg.get("human_discard_preference_bonus", 0.0))


def discard_preference_score(tile: int, hand: list[int], wildcard_enabled: bool = True) -> float:
    """Human-style discard priority for isolated low-value tiles.

    Higher means the tile is more reasonable to discard. This is intentionally
    lightweight and only rewards obvious basics: isolated honors, isolated
    terminals, and weak edge tiles. It avoids telling the model to discard
    valuable pairs, connected shapes, or live xiaoji.
    """

    if wildcard_enabled and tile == WILDCARD:
        return -4.0
    c = counts(hand)
    if c[tile] >= 2:
        return -0.8
    if tile in HONORS:
        return 1.4
    rank = tile % 9 + 1
    neighbor_count = _near_neighbor_count(tile, hand)
    if rank in {1, 9}:
        if neighbor_count == 0:
            return 1.2
        if neighbor_count == 1:
            return 0.45
        return -0.15
    if rank in {2, 8}:
        if neighbor_count == 0:
            return 0.75
        if neighbor_count == 1:
            return 0.2
        return -0.25
    if neighbor_count == 0:
        return 0.25
    return -0.35


def _standard_goal_score(hand: list[int], open_melds: int, wildcard_enabled: bool) -> float:
    shanten, shape_score = fast_hand_value(hand, open_melds=open_melds, wildcard_enabled=wildcard_enabled)
    edge_penalty = _edge_isolation_penalty(hand)
    # Standard hands are the default, easiest-to-complete direction. Keep this
    # score positive even when the hand is far away, otherwise rare high-fan
    # shapes get selected too aggressively in early turns.
    distance_score = max(0.0, 8.0 - float(shanten)) * 1.15
    return distance_score + shape_score * 0.04 + open_melds * 0.35 - edge_penalty


def _seven_pairs_goal_score(hand: list[int], wildcard_enabled: bool) -> float:
    c = counts(hand)
    wildcards = c[WILDCARD] if wildcard_enabled else 0
    if wildcard_enabled:
        c[WILDCARD] = 0
    pairs = sum(1 for n in c if n >= 2)
    near_pairs = sum(1 for n in c if n == 1)
    pair_waits = sum(max(0, 4 - n) for n in c if n == 1)
    missing_anchor = max(0, 4 - pairs - min(wildcards, near_pairs))
    score = pairs * 1.25 + min(wildcards, near_pairs) * 1.1 + pair_waits * 0.02
    score -= missing_anchor * 1.6
    if pairs + min(wildcards, near_pairs) < 4:
        score -= 2.0
    return score


def _flush_goal_score(all_tiles: list[int]) -> float:
    suit_counts = [0, 0, 0]
    honor_count = 0
    off_suit_count = 0
    for tile in all_tiles:
        if tile in HONORS:
            honor_count += 1
        else:
            suit = tile_suit(tile)
            if suit in {"m", "p", "s"}:
                suit_counts[{"m": 0, "p": 1, "s": 2}[suit]] += 1
    main = max(suit_counts)
    off_suit_count = sum(suit_counts) - main
    score = main * 0.9 + honor_count * 0.1 - off_suit_count * 1.55
    if main < 7:
        score -= 1.5
    return score


def _triplet_goal_score(all_tiles: list[int]) -> float:
    c = counts(all_tiles)
    pairs = sum(1 for n in c if n == 2)
    triplets = sum(1 for n in c if n >= 3)
    quads = sum(1 for n in c if n >= 4)
    return triplets * 1.8 + quads * 0.5 + pairs * 0.72


def _edge_isolation_penalty(hand: list[int]) -> float:
    c = counts(hand)
    penalty = 0.0
    for tile in hand:
        if tile in HONORS:
            continue
        rank = tile % 9
        if rank in {0, 8}:
            neighbors = 0
            for delta in (-2, -1, 1, 2):
                other = tile + delta
                if 0 <= other < 27 and tile_suit(other) == tile_suit(tile):
                    neighbors += c[other]
            if neighbors == 0:
                penalty += 0.2
            elif neighbors == 1:
                penalty += 0.08
    return penalty


def _near_neighbor_count(tile: int, hand: list[int]) -> int:
    if tile in HONORS:
        return 0
    return sum(
        hand.count(other)
        for other in (tile - 2, tile - 1, tile + 1, tile + 2)
        if 0 <= other < 27 and tile_suit(other) == tile_suit(tile)
    )
