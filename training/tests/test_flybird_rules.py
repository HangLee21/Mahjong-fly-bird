from mahjong_ai.env.actions import ACTION_KONG_ADDED, ACTION_KONG_CONCEALED, ACTION_PASS, ACTION_WIN
from mahjong_ai.rules.flybird import FlybirdRuleEngine, GameState, Meld, is_lanpai, is_seven_pairs, is_standard_win


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
