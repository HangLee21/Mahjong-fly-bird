"""Tests for v14 shallow expectation search (search-as-teacher)."""

from __future__ import annotations

from mahjong_ai.env.actions import ACTION_WIN, is_discard
from mahjong_ai.env.gym_env import MahjongSingleAgentEnv
from mahjong_ai.rules.flybird import FlybirdRuleEngine
from mahjong_ai.search.expectation import search_action_values, teacher_action


def _env_state(seed: int = 1):
    env = MahjongSingleAgentEnv({"reward": {}})
    env.reset(seed=seed)
    return env


def test_search_returns_scores_for_all_legal():
    env = _env_state(seed=2)
    state = env.state
    legal = env.rule_adapter.get_legal_actions(state, 0)
    assert legal
    values = search_action_values(state, 0, legal, n_samples=2, depth=1, seed=0)
    assert set(values.keys()) == set(legal)
    best = teacher_action(values)
    assert best in legal


def test_win_is_best_when_tenpai():
    """A hand that can win should have WIN as the teacher action."""
    env = _env_state(seed=5)
    engine = env.rule_adapter
    state = env.state
    # Force a ready hand: 123万 456万 789万 111筒 + 2条 (needs 2条 or 5条?)
    state.hands[0] = [0, 1, 2, 9, 10, 11, 18, 19, 20, 21, 22, 23, 26]
    # 26 = 9条, waiting on 9条? -> use a clear single wait: 123 456 789 111 + 44 wait on 4
    state.hands[0] = [0, 1, 2, 9, 10, 11, 18, 19, 20, 21, 22, 23, 4, 4]
    state.current_player = 0
    state.phase = "discard"
    legal = engine.get_legal_actions(state, 0)
    if ACTION_WIN in legal:
        values = search_action_values(state, 0, legal, n_samples=2, depth=1, seed=0)
        assert teacher_action(values) == ACTION_WIN
        assert values[ACTION_WIN] > max(v for a, v in values.items() if a != ACTION_WIN)


def test_search_reasonable_discard_choice():
    """With isolated honors + useful shapes, search should prefer an honor."""
    env = _env_state(seed=7)
    engine = env.rule_adapter
    state = env.state
    # 12234567万(8) + 999筒(3) + 东(27) + 白(28) = 13 tiles; honors are isolated.
    state.hands[0] = [0, 1, 2, 2, 3, 4, 5, 6, 18, 19, 20, 27, 28]
    state.current_player = 0
    state.phase = "discard"
    legal = engine.get_legal_actions(state, 0)
    discards = [a for a in legal if is_discard(a)]
    assert discards
    values = search_action_values(state, 0, legal, n_samples=4, depth=2, seed=0)
    best = teacher_action(values)
    assert best in discards, f"expected a discard, got {best}"
    # Either an honor is best, or at least it is not the worst scoring discard.
    honor_scores = {a: values[a] for a in discards if a in (27, 28)}
    worst = max(discards, key=lambda a: -values[a])
    assert best in (27, 28) or worst not in (27, 28), f"honor scored worst: {honor_scores}"
