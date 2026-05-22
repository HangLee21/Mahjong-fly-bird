from mahjong_ai.env.actions import ACTION_KONG_ADDED, ACTION_KONG_CONCEALED, ACTION_PASS, ACTION_PONG, ACTION_WIN
from mahjong_ai.rules.flybird import (
    FlybirdRuleEngine,
    GameState,
    Meld,
    is_four_xiaoji,
    is_lanpai,
    is_seven_pairs,
    is_standard_win,
    score_hand,
)


def test_standard_win_with_xiaoji_wildcard():
    tiles = [0, 1, 2, 3, 4, 5, 9, 10, 11, 27, 27, 27, 18, 33]
    assert is_standard_win(tiles, wildcard_enabled=True)


def test_seven_pairs_and_lanpai():
    assert is_seven_pairs([0, 0, 1, 1, 9, 9, 10, 10, 18, 18, 27, 27, 31, 31])
    assert is_lanpai([0, 3, 6, 10, 13, 16, 20, 23, 26, 27, 28, 29, 30, 31])


def test_discarded_xiaoji_disables_wildcard():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=1)
    state.hands[0] = [18, 0, 1, 2, 3, 4, 5, 9, 10, 11, 27, 27, 27, 33]
    next_state = engine.step(state, 0, 18)
    assert next_state.xiaoji_disabled


def test_concealed_kong_can_use_xiaoji_as_wildcard():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=2)
    state.hands[0] = [31, 31, 18, 18, 0, 1, 2, 3, 4, 5, 9, 10, 11, 27]
    assert ACTION_KONG_CONCEALED in engine.get_legal_actions(state, 0)


def test_base_hand_with_wildcard_cannot_ron():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=3)
    state.current_player = 1
    state.hands[1] = [0, 1, 2, 3, 4, 5, 9, 10, 11, 27, 27, 27, 18, 33]
    state.pending = None
    state = engine.step(state, 1, 33)
    # Player 2 gets first response to a hand that would only be bottom win.
    state.hands[2] = [0, 1, 2, 3, 4, 5, 9, 10, 11, 27, 27, 27, 18]
    legal = engine.get_legal_actions(state, 2)
    assert ACTION_WIN not in legal
    assert ACTION_PASS in legal


def test_added_kong_skips_ineligible_pong_melds():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=4)
    state.hands[0] = [6, 0, 1, 2, 3, 4, 7, 9, 10, 11, 27]
    state.melds[0] = [
        Meld("pong", [5, 5, 5]),
        Meld("pong", [6, 6, 6]),
    ]
    assert ACTION_KONG_ADDED in engine.get_legal_actions(state, 0)
    next_state = engine.step(state, 0, ACTION_KONG_ADDED)
    assert next_state.melds[0][0].type == "pong"
    assert next_state.melds[0][1].type == "kong"
    assert next_state.melds[0][1].added_from_pong
    assert not next_state.melds[0][1].concealed


def test_kong_pool_stays_two_tiles_after_kong_when_wall_available():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=40)
    state.hands[0] = [31, 31, 31, 31, 0, 1, 2, 3, 4, 5, 9, 10, 11, 27]
    before_wall = len(state.wall)
    assert len(state.kong_pool) == 2
    next_state = engine.step(state, 0, ACTION_KONG_CONCEALED)
    assert len(next_state.kong_pool) == 2
    assert len(next_state.wall) == before_wall - 1


def test_multi_ron_scores_all_winners():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=20)
    state.hands[0] = [31, 0, 1, 2, 3, 4, 5, 9, 10, 11, 27, 28, 29, 30]
    ready = [0, 1, 2, 3, 4, 5, 9, 10, 11, 27, 27, 27, 31]
    state.hands[1] = ready[:]
    state.hands[2] = ready[:]
    state.hands[3] = [0, 2, 4, 6, 8, 9, 11, 13, 15, 17, 27, 28, 29]
    state = engine.step(state, 0, 31)
    assert ACTION_WIN in engine.get_legal_actions(state, 1)
    state = engine.step(state, 1, ACTION_WIN)
    assert state.terminal
    assert state.winners == [1, 2]
    assert state.scores[1] > 0 and state.scores[2] > 0 and state.scores[0] < 0


def test_rob_kong_window_and_win():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=21)
    state.hands[0] = [5, 0, 1, 2, 3, 4, 6, 9, 10, 11, 27]
    state.melds[0] = [Meld("pong", [5, 5, 5])]
    state.hands[1] = [0, 1, 2, 5, 9, 10, 11, 12, 13, 14, 27, 27, 27]
    state.hands[2] = [0, 2, 4, 6, 8, 9, 11, 13, 15, 17, 27, 28, 29]
    state.hands[3] = state.hands[2][:]
    state = engine.step(state, 0, ACTION_KONG_ADDED)
    assert state.pending.kind == "rob_kong"
    assert ACTION_WIN in engine.get_legal_actions(state, 1)
    state = engine.step(state, 1, ACTION_WIN)
    assert state.terminal
    assert state.win_type == "rob_kong"
    assert "抢杠" in state.win_names


def test_kong_after_discard_adds_gang_shang_pao_name():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=22)
    state.last_kong_player = 0
    state.current_player = 0
    state.hands[0] = [31, 0, 1, 2, 3, 4, 5, 9, 10, 11, 27, 28, 29, 30]
    state.hands[1] = [0, 1, 2, 3, 4, 5, 9, 10, 11, 27, 27, 27, 31]
    state = engine.step(state, 0, 31)
    state = engine.step(state, 1, ACTION_WIN)
    assert "杠上炮" in state.win_names


def test_same_round_furiten_and_reject_pong():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=23)
    state.current_player = 0
    state.hands[0] = [31, 0, 1, 2, 3, 4, 5, 9, 10, 11, 27, 28, 29, 30]
    state.hands[1] = [0, 1, 2, 3, 4, 5, 9, 10, 11, 27, 27, 27, 31]
    state = engine.step(state, 0, 31)
    assert ACTION_WIN in engine.get_legal_actions(state, 1)
    state = engine.step(state, 1, ACTION_PASS)
    state.pending = type(state.pending)(discarder=2, tile=31, responders=[1])
    assert ACTION_WIN not in engine.get_legal_actions(state, 1)

    pong_state = engine.reset(seed=24)
    pong_state.current_player = 0
    pong_state.hands[0] = [5, 0, 1, 2, 3, 4, 6, 9, 10, 11, 27, 28, 29, 30]
    pong_state.hands[1] = [5, 5, 0, 1, 2, 3, 4, 6, 9, 10, 11, 27, 28]
    pong_state = engine.step(pong_state, 0, 5)
    assert ACTION_PONG in engine.get_legal_actions(pong_state, 1)
    pong_state = engine.step(pong_state, 1, ACTION_PASS)
    pong_state.pending = type(pong_state.pending)(discarder=2, tile=5, responders=[1])
    assert ACTION_PONG not in engine.get_legal_actions(pong_state, 1)


def test_ten_winds_and_thirteen_special_wins():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=25)
    state.current_player = 0
    state.special_discards[0] = [27, 28, 29, 30, 31, 32, 33, 27, 28]
    state.hands[0] = [29, 0, 1, 2]
    state = engine.step(state, 0, 29)
    assert state.terminal
    assert state.win_names == ["十风"]

    state = engine.reset(seed=26)
    state.current_player = 0
    state.special_discards[0] = [27, 28, 29, 30, 31, 32, 33, 0, 8, 9, 17, 18]
    state.hands[0] = [26, 0, 1, 2]
    state = engine.step(state, 0, 26)
    assert state.terminal
    assert state.win_names == ["十三幺有鸡"]


def test_four_xiaoji_is_immediate_special_win():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=41)
    state.hands[0] = [18, 18, 18, 18, 0, 1, 2, 3, 4, 5, 9, 10, 11, 27]
    assert is_four_xiaoji(state.hands[0], state.melds[0])
    assert ACTION_WIN in engine.get_legal_actions(state, 0)
    state = engine.step(state, 0, ACTION_WIN)
    assert state.terminal
    assert state.win_names == ["四小鸡"]
    assert state.win_points == 24.0


def test_lanpai_and_qixing_lanpai_score_without_menqing_stack():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=42)
    state.hands[0] = [0, 3, 6, 10, 13, 16, 20, 23, 26, 27, 28, 29, 30, 31]
    score = score_hand(state, 0, 31, self_draw=True)
    assert "烂牌" in score["names"]
    assert "门清自摸" not in score["names"]
    assert score["points"] == 4

    state.hands[0] = [0, 3, 6, 10, 13, 16, 20, 27, 28, 29, 30, 31, 32, 33]
    score = score_hand(state, 0, 33, self_draw=True)
    assert "七星烂牌" in score["names"]
    assert score["points"] == 8


def test_nested_qingyise_dadui_scores_as_capped_compound_fan():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=43)
    state.hands[0] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4]
    score = score_hand(state, 0, 4, self_draw=True)
    assert "清一色" in score["names"]
    assert "大对" in score["names"]
    assert score["fan"] == 3
    assert score["points"] == 8


def test_gangshang_flower_and_five_plum_score():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=44)
    state.last_draw_from_kong = True
    state.last_kong_player = 0
    state.hands[0] = [9, 10, 11, 12, 12, 12, 13, 13, 13, 14, 15, 16, 17, 17]
    score = score_hand(state, 0, 13, self_draw=True)
    assert "杠上开花" in score["names"]
    assert "五梅花" in score["names"]
    assert score["fan"] == 3
    assert score["points"] == 8
