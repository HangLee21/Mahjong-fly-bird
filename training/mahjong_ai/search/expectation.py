"""Shallow expectation search: score candidate actions by simulated draw streams.

AlphaZero-style "search as teacher": for each legal action we simulate plausible
future draws (sampled from the remaining-tile distribution) plus a simple
efficiency heuristic, and score the resulting hand quality. The action with the
best expected score becomes a teacher label for behavior cloning, giving the
policy dense decision-quality signal without any search cost at inference.

The simulation is hand-level only (no full game-state advancement): applying a
candidate action mutates a copy of the controlled hand / open-meld count, then
draw-discard cycles are scored with best_shanten / fast_hand_value. This keeps
the search cheap enough to run offline while data is collected.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from mahjong_ai.env.actions import (
    ACTION_CHOW_LEFT,
    ACTION_CHOW_MIDDLE,
    ACTION_CHOW_RIGHT,
    ACTION_KONG_ADDED,
    ACTION_KONG_CONCEALED,
    ACTION_KONG_EXPOSED,
    ACTION_PASS,
    ACTION_PONG,
    ACTION_WIN,
    N_TILE_TYPES,
    is_discard,
)
from mahjong_ai.rules.shanten import best_shanten, fast_hand_value

CHOW_OFFSETS = {
    ACTION_CHOW_LEFT: (1, 2),
    ACTION_CHOW_MIDDLE: (-1, 1),
    ACTION_CHOW_RIGHT: (-2, -1),
}


def _remaining_counts(state: Any, player_id: int) -> np.ndarray:
    """Per-tile unseen counts (own hand + all public discards/melds)."""
    seen = np.zeros(N_TILE_TYPES, dtype=np.float32)
    for tile in getattr(state, "hands", [[]] * 4)[player_id]:
        if 0 <= int(tile) < N_TILE_TYPES:
            seen[int(tile)] += 1.0
    for discards in getattr(state, "discards", []):
        for tile in discards:
            if 0 <= int(tile) < N_TILE_TYPES:
                seen[int(tile)] += 1.0
    for melds in getattr(state, "melds", []):
        for meld in melds:
            for tile in getattr(meld, "tiles", []):
                if 0 <= int(tile) < N_TILE_TYPES:
                    seen[int(tile)] += 1.0
    return np.clip(4.0 - seen, 0.0, 4.0)


def _sample_draw(rng: random.Random, remaining: np.ndarray) -> int | None:
    total = float(remaining.sum())
    if total <= 0.0:
        return None
    probs = remaining / total
    r = rng.random()
    acc = 0.0
    for tile in range(N_TILE_TYPES):
        acc += float(probs[tile])
        if r <= acc:
            return tile
    return int(np.argmax(probs))


def _apply_action(hand: list[int], open_melds: int, action: int, pending_tile: int | None) -> tuple[list[int], int]:
    """Apply a candidate action to a hand copy; return (new_hand, new_open_melds)."""
    if is_discard(action):
        if action in hand:
            hand = hand[:]
            hand.remove(action)
        return hand, open_melds
    if action in (ACTION_PASS,):
        return list(hand), open_melds
    if action == ACTION_WIN:
        return list(hand), open_melds
    if action == ACTION_PONG:
        hand = hand[:]
        if pending_tile is not None and hand.count(pending_tile) >= 2:
            hand.remove(pending_tile)
            hand.remove(pending_tile)
        return hand, open_melds + 1
    if action in CHOW_OFFSETS:
        hand = hand[:]
        d0, d1 = CHOW_OFFSETS[action]
        if pending_tile is not None:
            for need in (pending_tile + d0, pending_tile + d1):
                if need in hand:
                    hand.remove(need)
        return hand, open_melds + 1
    if action == ACTION_KONG_EXPOSED:
        hand = hand[:]
        if pending_tile is not None and hand.count(pending_tile) >= 3:
            hand = [t for t in hand if t != pending_tile]
        return hand, open_melds + 1
    if action in (ACTION_KONG_CONCEALED, ACTION_KONG_ADDED):
        hand = hand[:]
        # Concealed kong removes 4 of a kind; added kong removes 1 (completes a pong).
        counts: dict[int, int] = {}
        for t in hand:
            counts[t] = counts.get(t, 0) + 1
        quad = next((t for t, c in counts.items() if c == 4), None)
        if action == ACTION_KONG_CONCEALED and quad is not None:
            hand = [t for t in hand if t != quad]
        return hand, open_melds + 1
    return list(hand), open_melds


def _heuristic_discard(hand: list[int], open_melds: int, wild: bool, rng: random.Random) -> int | None:
    if not hand:
        return None
    base = best_shanten(hand, open_melds=open_melds, wildcard_enabled=wild)
    best_tiles: list[int] = []
    for tile in set(hand):
        trial = hand[:]
        trial.remove(tile)
        s = best_shanten(trial, open_melds=open_melds, wildcard_enabled=wild)
        if s < base:
            return tile
        if s == base:
            best_tiles.append(tile)
    if best_tiles:
        return rng.choice(best_tiles)
    return hand[0]


def _hand_score(hand: list[int], open_melds: int, wild: bool) -> float:
    shanten = best_shanten(hand, open_melds=open_melds, wildcard_enabled=wild)
    _, shape = fast_hand_value(hand, open_melds=open_melds, wildcard_enabled=wild)
    return -float(shanten) * 10.0 + min(float(shape), 80.0) * 0.1


def search_action_values(
    state: Any,
    player_id: int,
    legal_actions: list[int],
    *,
    n_samples: int = 4,
    depth: int = 2,
    seed: int = 0,
) -> dict[int, float]:
    """Expected hand-quality score for each legal action after simulated draws.

    Returns a dict {action_id: expected_score}. The teacher action is the one
    with the maximum score (callers may also inspect the full distribution).
    """
    rng = random.Random(seed)
    remaining = _remaining_counts(state, player_id)
    hand = list(getattr(state, "hands", [[]] * 4)[player_id])
    open_melds = len(getattr(state, "melds", [[]] * 4)[player_id])
    wild = not bool(getattr(state, "xiaoji_disabled", False))
    pending = getattr(state, "pending", None)
    pending_tile = None if pending is None else int(getattr(pending, "tile", -1))

    scores: dict[int, float] = {}
    for action in legal_actions:
        action = int(action)
        total = 0.0
        valid = 0
        for _ in range(n_samples):
            sim_hand, sim_open = _apply_action(hand, open_melds, action, pending_tile)
            if action == ACTION_WIN:
                total += 100.0  # winning is always best
                valid += 1
                continue
            for _d in range(depth):
                drawn = _sample_draw(rng, remaining)
                if drawn is None:
                    break
                sim_hand = sim_hand + [drawn]
                discard = _heuristic_discard(sim_hand, sim_open, wild, rng)
                if discard is None:
                    break
                sim_hand = sim_hand[:]
                sim_hand.remove(discard)
            total += _hand_score(sim_hand, sim_open, wild)
            valid += 1
        scores[action] = total / max(1, valid)
    return scores


def teacher_action(scores: dict[int, float]) -> int:
    """Argmax of expected scores: the search's recommended move."""
    return max(scores, key=scores.get)
