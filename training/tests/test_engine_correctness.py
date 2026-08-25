"""Correctness differential tests for the engine optimizations.

Each optimized function is checked against an independent naive reference so the
speedups (shallow copies, custom __deepcopy__, memoized meld search, parity
prune) are provably behavior-identical.
"""
from __future__ import annotations

import copy
import random

import pytest

from mahjong_ai.rules.flybird import (
    N_TILE_TYPES,
    WILDCARD,
    FlybirdRuleEngine,
    Meld,
    PendingClaim,
    _can_form_melds,
    is_standard_win,
    is_win_shape,
    tile_rank,
    tile_suit,
)


# --------------------------------------------------------------------------
# Naive references (the pre-optimization algorithms).
# --------------------------------------------------------------------------
def _ref_can_form_melds(c: list[int], wildcards: int) -> bool:
    try:
        first = next(i for i, v in enumerate(c) if v)
    except StopIteration:
        return wildcards % 3 == 0

    need = max(0, 3 - c[first])
    if need <= wildcards:
        used = min(3, c[first])
        c[first] -= used
        if _ref_can_form_melds(c, wildcards - need):
            c[first] += used
            return True
        c[first] += used

    suit = tile_suit(first)
    rank = tile_rank(first)
    if suit != "z" and rank is not None and rank <= 7:
        seq = [first, first + 1, first + 2]
        branch = c[:]
        missing = 0
        for t in seq:
            if branch[t] > 0:
                branch[t] -= 1
            else:
                missing += 1
        if missing <= wildcards and all(tile_suit(t) == suit for t in seq):
            if _ref_can_form_melds(branch, wildcards - missing):
                return True
    return False


def _ref_standard_win(tiles: list[int], wildcard_enabled: bool = True) -> bool:
    c = [0] * N_TILE_TYPES
    for t in tiles:
        c[t] += 1
    wildcards = c[WILDCARD] if wildcard_enabled else 0
    if wildcard_enabled:
        c[WILDCARD] = 0
    for pair_tile in range(N_TILE_TYPES):
        natural = c[pair_tile]
        for use_wild in range(0, min(2, wildcards) + 1):
            if natural + use_wild >= 2:
                branch = c[:]
                branch[pair_tile] = max(0, branch[pair_tile] - (2 - use_wild))
                if _ref_can_form_melds(branch, wildcards - use_wild):
                    return True
    return wildcards >= 2 and _ref_can_form_melds(c[:], wildcards - 2)


def _random_hand(rng: random.Random, n: int | None = None) -> list[int]:
    n = n if n is not None else rng.choice([2, 5, 8, 11, 13, 14])
    return [rng.randrange(N_TILE_TYPES) for _ in range(n)]


@pytest.mark.parametrize("seed", range(5))
def test_can_form_melds_matches_reference(seed: int):
    rng = random.Random(seed)
    for _ in range(2000):
        hand = _random_hand(rng)
        c = [0] * N_TILE_TYPES
        for t in hand:
            c[t] += 1
        for wildcards in range(0, 5):
            assert _can_form_melds(c[:], wildcards) == _ref_can_form_melds(c[:], wildcards), (hand, wildcards)


@pytest.mark.parametrize("seed", range(5))
def test_standard_win_matches_reference(seed: int):
    rng = random.Random(seed)
    for _ in range(3000):
        hand = _random_hand(rng)
        for wild_enabled in (True, False):
            got = is_standard_win(hand, wild_enabled)
            ref = _ref_standard_win(hand, wild_enabled)
            assert got == ref, (hand, wild_enabled)


@pytest.mark.parametrize("seed", range(3))
def test_clone_state_deep_equals_and_independent(seed: int):
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=seed)
    # Deterministic state with every mutable nesting populated, especially a
    # Meld (mutated in place by the added-kong path) and a PendingClaim.
    state.melds[0] = [Meld("pong", [5, 5, 5], from_player=2)]
    state.pending = PendingClaim(discarder=2, tile=5, responders=[0])
    state.discards = [[1, 2], [3], [], [4]]
    state.public_events = [{"type": "discard", "player": 0, "tile": 1, "step": 1, "wall": 100}]
    state.same_round_furiten[0] = {7}
    state.reject_pong_tiles[1] = {8}
    state.special_discards[2] = [9]
    state.hands[0] = [0, 1, 2]

    clone = engine.clone_state(state)
    reference = copy.deepcopy(state)

    assert clone == reference, "optimized clone must deep-equal reference deepcopy"
    assert clone is not state

    # Mutate every mutable nesting of the clone; the original must be untouched.
    clone.melds[0][0].type = "kong"
    clone.melds[0][0].tiles.append(99)
    clone.melds[0].append(Meld("pong", [1, 1, 1]))
    clone.hands[0].append(99)
    clone.discards[1].append(99)
    clone.public_events.append({"type": "zzz", "player": 9, "tile": 99, "step": 9, "wall": 9})
    clone.same_round_furiten[0].add(99)
    clone.reject_pong_tiles[1].add(99)
    clone.special_discards[2].append(99)
    clone.pending.responders.append(99)

    assert state.melds[0][0].type == "pong", "clone meld mutation leaked into original"
    assert state.melds[0][0].tiles == [5, 5, 5]
    assert len(state.melds[0]) == 1
    assert 99 not in state.hands[0]
    assert 99 not in state.discards[1]
    assert state.public_events[-1].get("type") != "zzz"
    assert 99 not in state.same_round_furiten[0]
    assert 99 not in state.reject_pong_tiles[1]
    assert 99 not in state.special_discards[2]
    assert 99 not in state.pending.responders
    assert reference == state, "reference deepcopy must still equal original after clone mutation"


def test_get_public_info_is_value_accurate():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=7)
    state.discards[0] = [1, 2, 3]
    state.melds[1] = [Meld("pong", [5, 5, 5], from_player=2)]
    public = engine.get_public_info(state)
    assert public["discards"][0] == [1, 2, 3]
    assert public["discards"][0] is not state.discards[0]  # outer list copied
    assert public["melds"][1][0].tiles == [5, 5, 5]
    assert public["scores"] == list(state.scores)
    assert public["public_events"] == list(state.public_events)


@pytest.mark.parametrize("seed", [11, 23, 47])
def test_full_game_clone_matches_deepcopy_every_step(seed: int):
    """End-to-end: clone_state must equal copy.deepcopy at every step of real play."""
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=seed)
    rng = random.Random(seed)
    for _ in range(800):
        if state.terminal:
            break
        clone = engine.clone_state(state)
        assert clone == copy.deepcopy(state), f"clone mismatch at step {state.step_count}"
        assert clone is not state
        player = engine.get_current_player(state)
        legal = engine.get_legal_actions(state, player)
        if not legal:
            break
        state = engine.step(state, player, rng.choice(legal))
