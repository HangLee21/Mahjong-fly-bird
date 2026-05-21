from mahjong_ai.env.actions import ACTION_KONG_CONCEALED
from mahjong_ai.env.reward import compute_reward
from mahjong_ai.rules.flybird import FlybirdRuleEngine, WILDCARD


def test_xiaoji_discard_shaping_penalty():
    engine = FlybirdRuleEngine()
    prev_state = engine.reset(seed=11)
    next_state = engine.clone_state(prev_state)
    reward = compute_reward(
        prev_state,
        next_state,
        0,
        engine,
        {"discard_live_xiaoji_penalty": 0.04},
        action=WILDCARD,
    )
    assert reward == -0.04


def test_concealed_kong_shaping_bonus():
    engine = FlybirdRuleEngine()
    prev_state = engine.reset(seed=12)
    next_state = engine.clone_state(prev_state)
    reward = compute_reward(
        prev_state,
        next_state,
        0,
        engine,
        {"concealed_kong_bonus": 0.004},
        action=ACTION_KONG_CONCEALED,
    )
    assert reward == 0.004


def test_shanten_improvement_reward():
    engine = FlybirdRuleEngine()
    prev_state = engine.reset(seed=13)
    next_state = engine.clone_state(prev_state)
    prev_state.hands[0] = [0, 1, 3, 4, 5, 9, 10, 11, 27, 27, 31, 32, 33, 8]
    next_state.hands[0] = [0, 1, 2, 3, 4, 5, 9, 10, 11, 27, 27, 31, 32]
    reward = compute_reward(
        prev_state,
        next_state,
        0,
        engine,
        {"shanten_improvement_bonus": 0.01},
        action=8,
    )
    assert reward > 0
