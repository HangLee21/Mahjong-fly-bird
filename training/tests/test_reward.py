from mahjong_ai.env.actions import ACTION_KONG_ADDED, ACTION_KONG_CONCEALED
from mahjong_ai.env.reward import compute_reward, discard_preference_score, hand_goal_scores_for_tiles
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


def test_added_kong_can_be_rewarded_as_concealed_style_kong():
    engine = FlybirdRuleEngine()
    prev_state = engine.reset(seed=17)
    next_state = engine.clone_state(prev_state)
    reward = compute_reward(
        prev_state,
        next_state,
        0,
        engine,
        {
            "added_kong_as_concealed": True,
            "added_kong_as_concealed_bonus": 0.004,
            "added_kong_penalty": 0.01,
        },
        action=ACTION_KONG_ADDED,
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


def test_self_draw_score_reward_is_three_sided():
    engine = FlybirdRuleEngine()
    prev_state = engine.reset(seed=14)
    next_state = engine.clone_state(prev_state)
    next_state.terminal = True
    next_state.win_type = "self_draw"
    next_state.winners = [0]
    next_state.scores = [3.0, -1.0, -1.0, -1.0]
    reward = compute_reward(prev_state, next_state, 0, engine, {"score_scale": 10.0})
    assert reward == 0.3


def test_hand_goal_shaping_rewards_flush_cleanup():
    engine = FlybirdRuleEngine()
    prev_state = engine.reset(seed=15)
    next_state = engine.clone_state(prev_state)
    prev_state.hands[0] = [0, 1, 2, 3, 4, 5, 9, 10, 11, 18, 19, 27, 28, 29]
    next_state.hands[0] = [0, 1, 2, 3, 4, 5, 10, 11, 18, 19, 27, 28, 29]
    reward = compute_reward(
        prev_state,
        next_state,
        0,
        engine,
        {"hand_goal_mode": "best_delta", "hand_goal_improvement_bonus": 0.002},
        action=9,
    )
    assert reward > 0


def test_committed_hand_goal_limits_uncommitted_shape_reward():
    engine = FlybirdRuleEngine()
    prev_state = engine.reset(seed=16)
    next_state = engine.clone_state(prev_state)
    prev_state.hands[0] = [0, 0, 1, 1, 2, 2, 9, 10, 11, 18, 19, 27, 28, 29]
    next_state.hands[0] = [0, 1, 2, 9, 10, 11, 18, 19, 20, 21, 22, 27, 28]
    reward = compute_reward(
        prev_state,
        next_state,
        0,
        engine,
        {
            "hand_goal_mode": "committed",
            "hand_goal_improvement_bonus": 0.002,
            "hand_goal_regression_penalty": 0.001,
            "hand_goal_switch_penalty": 0.005,
        },
        action=29,
    )
    assert reward < 0.001


def test_goal_scores_do_not_overselect_seven_pairs_too_early():
    hand = [4, 8, 12, 16, 19, 25, 28, 28, 29, 30, 31, 32, 32, 33]
    scores = hand_goal_scores_for_tiles(hand)
    target = max(range(len(scores)), key=lambda i: scores[i])
    assert target != 1


def test_human_discard_preference_values_isolated_tiles():
    hand = [0, 4, 9, 13, 18, 22, 27, 28, 29, 31, 31, 32, 33]
    assert discard_preference_score(27, hand) > discard_preference_score(31, hand)
    assert discard_preference_score(0, hand) > discard_preference_score(4, hand)
    assert discard_preference_score(18, hand, wildcard_enabled=True) < 0
