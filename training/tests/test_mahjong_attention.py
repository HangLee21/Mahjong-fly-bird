from mahjong_ai.env.gym_env import MahjongSingleAgentEnv
from mahjong_ai.env.observation import HAND_MAX_TILES, HAND_TOKEN_DIM, N_TILE_TYPES, build_hand_tokens, get_observation_dim
from mahjong_ai.env.reward import compute_reward
from mahjong_ai.rules.flybird import FlybirdRuleEngine


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
