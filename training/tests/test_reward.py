from mahjong_ai.env.actions import ACTION_KONG_ADDED, ACTION_KONG_CONCEALED
from mahjong_ai.env.actions import ACTION_CHOW_RIGHT, ACTION_PASS, ACTION_PONG
from mahjong_ai.env.reward import (
    _discard_taatsu_strength,
    _taatsu_wait_type_count,
    claim_decision_reward,
    compute_reward,
    discard_danger_score,
    discard_efficiency_reward,
    discard_preference_score,
    discard_value_order_reward,
    hand_goal_scores_for_tiles,
)
from mahjong_ai.rules.flybird import FlybirdRuleEngine, WILDCARD
from mahjong_ai.rules.flybird import PendingClaim


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


def test_discard_danger_score_is_safe_when_all_copies_visible():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=19)
    state.discards = [[5, 5], [5], [5], []]
    state.melds = [[], [], [], []]
    assert discard_danger_score(state, 5, 0) == 0.0


def test_discard_danger_score_live_middle_tile_is_dangerous():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=19)
    state.discards = [[], [], [5], []]
    state.melds = [[], [], [], []]
    assert discard_danger_score(state, 5, 0) == 0.75


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


def test_discard_efficiency_penalizes_breaking_completed_sequence():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=18)
    state.hands[0] = [5, 6, 12, 13, 13, 16, 17, 18, 21, 22, 23]
    bad = discard_efficiency_reward(
        state,
        0,
        23,
        {
            "discard_best_shanten_bonus": 0.002,
            "discard_miss_best_shanten_penalty": 0.012,
            "discard_break_meld_penalty": 0.01,
        },
    )
    better = discard_efficiency_reward(
        state,
        0,
        12,
        {
            "discard_best_shanten_bonus": 0.002,
            "discard_miss_best_shanten_penalty": 0.012,
            "discard_break_meld_penalty": 0.01,
        },
    )
    assert bad < 0
    assert better > bad


def test_discard_efficiency_penalizes_breaking_useful_taatsu():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=19)
    state.hands[0] = [1, 2, 5, 9, 13, 13, 16, 17, 18, 21, 22, 23]
    bad = discard_efficiency_reward(
        state,
        0,
        1,
        {
            "discard_best_shanten_bonus": 0.002,
            "discard_miss_best_shanten_penalty": 0.006,
            "discard_break_taatsu_penalty": 0.004,
            "discard_break_good_taatsu_penalty": 0.008,
        },
    )
    better = discard_efficiency_reward(
        state,
        0,
        5,
        {
            "discard_best_shanten_bonus": 0.002,
            "discard_miss_best_shanten_penalty": 0.006,
            "discard_break_taatsu_penalty": 0.004,
            "discard_break_good_taatsu_penalty": 0.008,
        },
    )
    assert bad < better


def test_taatsu_wait_types_rank_open_waits_above_edge_and_closed_waits():
    assert _taatsu_wait_type_count(1, 2) == 2  # 2万3万 waits on 1万/4万
    assert _taatsu_wait_type_count(4, 5) == 2  # 5万6万 waits on 4万/7万
    assert _taatsu_wait_type_count(25, 26) == 1  # 8条9条 only waits on 7条
    assert _taatsu_wait_type_count(21, 23) == 1  # 4条6条 only waits on 5条


def test_overcomplete_hand_prefers_cutting_low_ukeire_shape():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=21)
    state.hands[0] = [1, 2, 4, 5, 12, 13, 13, 13, 14, 18, 21, 23, 25, 26]
    cfg = {
        "discard_best_shanten_bonus": 0.002,
        "discard_miss_best_shanten_penalty": 0.012,
        "discard_break_taatsu_penalty": 0.008,
        "discard_break_good_taatsu_penalty": 0.018,
        "discard_near_ready_good_taatsu_penalty": 0.028,
        "discard_weak_taatsu_bonus": 0.006,
    }
    cut_open_wait = discard_efficiency_reward(state, 0, 1, cfg)
    cut_edge_wait = discard_efficiency_reward(state, 0, 25, cfg)
    cut_isolated_terminal = discard_efficiency_reward(state, 0, 18, cfg)

    assert _discard_taatsu_strength(state.hands[0], 1) == 2
    assert _discard_taatsu_strength(state.hands[0], 4) == 2
    assert _discard_taatsu_strength(state.hands[0], 21) == 1
    assert _discard_taatsu_strength(state.hands[0], 25) == 1
    assert _discard_taatsu_strength(state.hands[0], 18) == 0
    assert cut_open_wait < cut_edge_wait
    assert cut_open_wait < cut_isolated_terminal


def test_discarding_instead_of_available_concealed_kong_is_penalized():
    engine = FlybirdRuleEngine()
    prev_state = engine.reset(seed=20)
    next_state = engine.clone_state(prev_state)
    prev_state.hands[0] = [0, 0, 0, 0, 1, 2, 4, 5, 6, 9, 10, 11, 18, 19]
    next_state.hands[0] = [0, 0, 0, 0, 1, 2, 4, 5, 6, 9, 10, 11, 19]
    reward = compute_reward(
        prev_state,
        next_state,
        0,
        engine,
        {"discard_over_kong_penalty": 0.003, "discard_over_kong_ready_penalty": 0.006},
        action=18,
    )
    assert reward < 0


def test_value_order_prefers_isolated_honor_over_connected_suit_tile():
    hand = [0, 1, 7, 10, 10, 14, 15, 18, 19, 23, 24, 27, 31, 33]
    cfg = {
        "isolated_honor_discard_bonus": 0.006,
        "discard_connected_suit_over_honor_penalty": 0.008,
        "discard_pair_over_honor_penalty": 0.01,
    }
    honor = discard_value_order_reward(33, hand, True, cfg)
    connected = discard_value_order_reward(1, hand, True, cfg)
    pair = discard_value_order_reward(10, hand, True, cfg)
    assert honor > 0
    assert connected < 0
    assert pair < connected


def test_value_order_penalizes_breaking_only_pair():
    hand = [0, 1, 7, 10, 10, 14, 15, 18, 19, 23, 24, 27, 29, 33]
    cfg = {"discard_pair_penalty": 0.004, "discard_only_pair_penalty": 0.01}
    assert discard_value_order_reward(10, hand, True, cfg) == -0.01


def test_claim_reward_rewards_improving_claim_and_penalizes_pass():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=22)
    state.current_player = 0
    state.pending = PendingClaim(discarder=3, tile=3, responders=[0])
    state.hands[0] = [1, 2, 4, 5, 12, 13, 13, 18, 21, 23, 25, 26, 28]
    cfg = {
        "claim_improvement_bonus": 0.018,
        "claim_same_penalty": 0.01,
        "pass_improving_claim_penalty": 0.018,
        "pass_same_claim_bonus": 0.004,
        "pass_non_improving_claim_bonus": 0.006,
    }
    assert claim_decision_reward(state, 0, ACTION_CHOW_RIGHT, cfg) > 0
    assert claim_decision_reward(state, 0, ACTION_PASS, cfg) < 0


def test_claim_reward_rewards_improving_pong_over_pass():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=23)
    state.current_player = 0
    state.pending = PendingClaim(discarder=3, tile=1, responders=[0])
    state.hands[0] = [1, 1, 2, 3, 4, 12, 13, 14, 18, 21, 22, 23, 28]
    cfg = {
        "claim_improvement_bonus": 0.018,
        "pass_improving_claim_penalty": 0.018,
    }
    assert claim_decision_reward(state, 0, ACTION_PONG, cfg) > 0
    assert claim_decision_reward(state, 0, ACTION_PASS, cfg) < 0
