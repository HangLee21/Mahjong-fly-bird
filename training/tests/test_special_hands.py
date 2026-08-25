"""Special-hand win-shape coverage, cross-checked against the rulebook (v1.5).

The rulebook (第4节) lists these win SHAPES: 底和(标准) / 小七对 / 烂牌(七星) /
十风 / 十三幺(有·无鸡) / 四小鸡. Every shape must be detected, and the
optimized meld-search must agree with a naive reference for the shapes it feeds.
"""
from __future__ import annotations

import random

import pytest

from mahjong_ai.rules.flybird import (
    HONORS,
    WILDCARD,
    FlybirdRuleEngine,
    PendingClaim,
    is_four_xiaoji,
    is_lanpai,
    is_seven_pairs,
    is_standard_win,
    is_win_shape,
)
from mahjong_ai.env.actions import ACTION_CHOW_LEFT, ACTION_CHOW_MIDDLE, ACTION_CHOW_RIGHT, ACTION_PONG


# --------------------------------------------------------------------------
# Naive references straight from the rulebook text.
# --------------------------------------------------------------------------
def _ref_seven_pairs(tiles: list[int], wildcard_enabled: bool = True) -> bool:
    if len(tiles) != 14:
        return False
    c = [0] * 34
    for t in tiles:
        c[t] += 1
    wild = c[WILDCARD] if wildcard_enabled else 0
    if wildcard_enabled:
        c[WILDCARD] = 0
    pairs = 0
    singles = 0
    for n in c:
        pairs += n // 2
        singles += n % 2
    if singles > wild:
        return False
    pairs += singles
    wild -= singles
    pairs += wild // 2
    return pairs >= 7


def _ref_lanpai(tiles: list[int], wildcard_enabled: bool = True) -> bool:
    natural = [t for t in tiles if not (wildcard_enabled and t == WILDCARD)]
    if any(natural.count(t) > 1 for t in set(natural)):
        return False
    honors = {t for t in natural if t in HONORS}
    if len(honors) < 5:
        return False
    for suit_start in (0, 9, 18):
        ranks = sorted(t - suit_start + 1 for t in natural if suit_start <= t < suit_start + 9)
        for i, a in enumerate(ranks):
            for b in ranks[i + 1:]:
                if abs(a - b) not in (3, 6):
                    return False
    return True


@pytest.mark.parametrize("seed", range(4))
def test_seven_pairs_matches_reference(seed: int):
    rng = random.Random(seed)
    for _ in range(3000):
        hand = [rng.randrange(34) for _ in range(14)]
        for wild in (True, False):
            assert is_seven_pairs(hand, wild) == _ref_seven_pairs(hand, wild), (hand, wild)


@pytest.mark.parametrize("seed", range(4))
def test_lanpai_matches_reference(seed: int):
    rng = random.Random(seed)
    for _ in range(3000):
        hand = [rng.randrange(34) for _ in range(14)]
        for wild in (True, False):
            assert is_lanpai(hand, wild) == _ref_lanpai(hand, wild), (hand, wild)


# --------------------------------------------------------------------------
# Concrete rulebook examples: every win shape must be detected.
# --------------------------------------------------------------------------
def test_standard_win_shape_detected():
    # 4 melds + pair (底和).
    hand = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12]
    assert is_standard_win(hand)
    assert is_win_shape(hand)


def test_seven_pairs_detected():
    hand = [0, 0, 1, 1, 9, 9, 10, 10, 18, 18, 27, 27, 31, 31]
    assert is_seven_pairs(hand)
    assert is_win_shape(hand)
    assert not is_standard_win(hand)


def test_seven_pairs_dragon_back_is_still_seven_pairs():
    # 小七对龙背: two identical pairs = four of a kind.
    hand = [0, 0, 0, 0, 1, 1, 9, 9, 10, 10, 27, 27, 31, 31]
    assert is_seven_pairs(hand)
    assert is_win_shape(hand)


def test_lanpai_detected():
    # 147万 258条 369饼 + 东南西北中 (5 honors).
    hand = [0, 3, 6, 10, 13, 16, 20, 23, 26, 27, 28, 29, 30, 31]
    assert is_lanpai(hand)
    assert is_win_shape(hand)
    assert not is_standard_win(hand)
    assert not is_seven_pairs(hand)


def test_qixing_lanpai_detected():
    # 烂牌 with all 7 honors (七星烂牌).
    hand = [0, 3, 6, 10, 13, 16, 20, 27, 28, 29, 30, 31, 32, 33]
    assert is_lanpai(hand)
    assert is_win_shape(hand)


def test_four_xiaoji_detected():
    assert is_four_xiaoji([18, 18, 18, 18])
    assert not is_four_xiaoji([18, 18, 18, 5])


def test_ten_winds_special_win():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=1)
    state.special_discards[0] = [27, 28, 29, 30, 31, 32, 33, 27, 28, 29]
    state.discarded_non_special[0] = False
    assert engine._special_win_name(state, 0) == "十风"
    assert engine._can_special_self_win(state, 0)


def test_thirteen_orphans_special_win():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=1)
    # 13 distinct honor/terminal tiles including the xiaoji (1条) -> 有鸡 (4分).
    state.special_discards[0] = [27, 28, 29, 30, 31, 32, 33, 0, 8, 9, 17, 18, 26]
    state.discarded_non_special[0] = False
    assert engine._special_win_name(state, 0) == "十三幺有鸡"
    assert engine._can_special_self_win(state, 0)


def test_thirteen_orphans_no_wildcard():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=1)
    # 13 special tiles with no xiaoji -> 无鸡 (8分).
    state.special_discards[0] = [27, 28, 29, 30, 31, 32, 33, 0, 8, 9, 17, 26, 27]
    state.discarded_non_special[0] = False
    assert engine._special_win_name(state, 0) == "十三幺无鸡"


def test_non_win_shapes_are_rejected():
    assert not is_win_shape([0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    assert not is_seven_pairs([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7])
    assert not is_lanpai([0, 0, 3, 6, 10, 13, 16, 20, 23, 26, 27, 28, 29, 30])


# --------------------------------------------------------------------------
# Xuanwei wildcard-disable rules (rulebook §3.1).
# --------------------------------------------------------------------------
def test_chow_xiaoji_as_natural_disables_wildcard():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=9)
    state.pending = PendingClaim(discarder=3, tile=WILDCARD, responders=[0])  # 上家打 1条
    state.hands[0] = [19, 20, 5, 9, 13, 13, 16, 17, 21, 22, 23, 26, 27]  # 2条 3条
    assert ACTION_CHOW_LEFT in engine.get_legal_actions(state, 0)  # 1条2条3条 natural chow
    next_state = engine.step(state, 0, ACTION_CHOW_LEFT)
    assert next_state.xiaoji_disabled


def test_pong_xiaoji_as_natural_disables_wildcard():
    # Pong-ing the 1 bamboo treats the xiaoji as its natural 1条 and disables
    # the wildcard, just like chow.
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=9)
    state.pending = PendingClaim(discarder=1, tile=WILDCARD, responders=[0])
    state.hands[0] = [18, 18, 5, 9, 13, 13, 16, 17, 21, 22, 23, 26, 27]
    assert ACTION_PONG in engine.get_legal_actions(state, 0)
    next_state = engine.step(state, 0, ACTION_PONG)
    assert next_state.xiaoji_disabled


def test_wildcard_cannot_substitute_in_chow():
    # Rulebook: cannot use 三万 + 小鸡(代四万) to chow 上家's 五万.
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=9)
    state.pending = PendingClaim(discarder=3, tile=4, responders=[0])  # 上家打 5万
    state.hands[0] = [2, 18, 9, 10, 11, 27, 28, 29, 30, 31, 32, 33, 26]  # 3万 + 小鸡
    legal = engine.get_legal_actions(state, 0)
    assert ACTION_CHOW_LEFT not in legal
    assert ACTION_CHOW_MIDDLE not in legal
    assert ACTION_CHOW_RIGHT not in legal
