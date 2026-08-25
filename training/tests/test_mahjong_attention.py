from mahjong_ai.env.gym_env import MahjongSingleAgentEnv
from mahjong_ai.env.observation import HAND_MAX_TILES, HAND_TOKEN_DIM, N_TILE_TYPES, build_hand_tokens, get_observation_dim
from mahjong_ai.env.actions import ACTION_CHOW_LEFT
from mahjong_ai.env.reward import claim_decision_reward, compute_reward
from mahjong_ai.rules.flybird import FlybirdRuleEngine, PendingClaim, WILDCARD


def test_hand_tokens_encode_hand_with_padding():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=7)
    state.hands[0] = [0, 1, 2, 18, 27, 33]
    tokens, mask = build_hand_tokens(engine, state, 0)
    assert tokens.shape == (HAND_MAX_TILES, HAND_TOKEN_DIM)
    assert mask.shape == (HAND_MAX_TILES,)
    assert int(mask.sum()) == 6  # 6 real tiles, rest padded
    # tile one-hot: first token is tile 0
    assert tokens[0, 0] == 1.0
    # xiaoji flag (tile 18) at token index 3
    assert tokens[3, N_TILE_TYPES + 1] == 1.0
    # honor flag (tile 27) at token index 4
    assert tokens[4, N_TILE_TYPES + 2] == 1.0


def test_hand_goal_appended_to_static_dim():
    cfg = {"observation": {"include_hand": True, "include_table": True}, "obs_include_hand_goal": True, "reward": {}}
    env = MahjongSingleAgentEnv(cfg)
    obs, _ = env.reset(seed=11)
    assert obs["static"].shape[0] == get_observation_dim(cfg) == 398  # 394 + 4 goal scores
    assert "hand" in obs and "hand_mask" in obs and "table" in obs


def test_win_fan_scale_rewards_bigger_hands_more():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=13)
    state.terminal = True
    state.winners = [0]
    state.win_type = "self_draw"
    # Give a flat win bonus with fan scaling; fan computed from the actual hand.
    reward = compute_reward(state, state, 0, engine, {"terminal_win_bonus": 0.1, "win_fan_scale": 1.0, "score_scale": 1.0})
    # flat part is 0.1, fan-scaled part adds terminal_win_bonus * fan.
    assert reward >= 0.1


def test_chowing_live_wildcard_is_penalized():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=5)
    state.pending = PendingClaim(discarder=3, tile=WILDCARD, responders=[0])
    # 1 bamboo + 2 bamboo + 3 bamboo chow (CHOW_LEFT uses tile+1, tile+2).
    state.hands[0] = [19, 20, 5, 9, 13, 13, 16, 17, 21, 22, 23, 26, 27]
    penalized = claim_decision_reward(state, 0, ACTION_CHOW_LEFT, {"claim_1tiao_penalty": 0.03})
    plain = claim_decision_reward(state, 0, ACTION_CHOW_LEFT, {"claim_1tiao_penalty": 0.0})
    assert penalized <= plain - 0.029
    assert penalized < 0.0


def test_ponging_live_wildcard_is_penalized():
    # Pong-ing the natural 1条 also disables the wildcard, so it is penalized.
    from mahjong_ai.env.actions import ACTION_PONG

    engine = FlybirdRuleEngine()
    state = engine.reset(seed=6)
    state.pending = PendingClaim(discarder=3, tile=WILDCARD, responders=[0])
    state.hands[0] = [18, 18, 5, 9, 13, 13, 16, 17, 21, 22, 23, 26, 27]
    penalized = claim_decision_reward(state, 0, ACTION_PONG, {"claim_1tiao_penalty": 0.03})
    plain = claim_decision_reward(state, 0, ACTION_PONG, {"claim_1tiao_penalty": 0.0})
    assert penalized < plain
    assert penalized < 0.0
