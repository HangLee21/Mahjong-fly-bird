import numpy as np

from mahjong_ai.env.gym_env import MahjongSingleAgentEnv
from mahjong_ai.env.observation import get_observation_dim
from mahjong_ai.inference.predictor import MahjongPredictor
from mahjong_ai.rules.flybird import FlybirdRuleEngine


def test_observation_shape_and_private_view():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=5)
    from mahjong_ai.env.observation import build_observation

    obs0 = build_observation(engine, state, 0, {})
    obs1 = build_observation(engine, state, 1, {})
    assert obs0.shape == (get_observation_dim({}),)
    assert np.isfinite(obs0).all()
    assert not np.array_equal(obs0[:34], obs1[:34])


def test_gym_env_runs_steps():
    env = MahjongSingleAgentEnv({"opponent_agent": "random", "max_steps_per_game": 120})
    obs, info = env.reset(seed=10)
    assert obs.shape == env.observation_space.shape
    assert info["legal_actions"]
    for _ in range(10):
        action = info["legal_actions"][0]
        obs, _, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        if terminated or truncated:
            break
        assert info["legal_actions"]


def test_env_auto_skips_forced_pass():
    env = MahjongSingleAgentEnv({"opponent_agent": "random", "max_steps_per_game": 120})
    _, info = env.reset(seed=10)
    assert info["legal_actions"] != [100]


def test_predictor_fallback_returns_legal_action():
    predictor = MahjongPredictor()
    obs = np.zeros(get_observation_dim({}), dtype=np.float32)
    result = predictor.predict(obs, [3, 100])
    assert result["action"] in [3, 100]
    assert result["fallback_used"]
