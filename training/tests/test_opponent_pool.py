import numpy as np

from mahjong_ai.agents.heuristic_agent import HeuristicAgent, WinFirstAgent
from mahjong_ai.agents.opponent_pool import OpponentPool
from mahjong_ai.agents.random_agent import RandomAgent
from mahjong_ai.env.gym_env import MahjongSingleAgentEnv


def test_opponent_pool_samples_table():
    pool = OpponentPool(
        {
            "members": [
                {"kind": "heuristic", "weight": 1.0},
                {"kind": "win_first", "weight": 1.0},
                {"kind": "random", "weight": 1.0},
            ]
        },
        seed=7,
    )
    table = pool.sample_table(controlled_player=0)
    assert table[0] is None
    assert all(agent is not None for agent in table[1:])
    assert all(isinstance(agent, (HeuristicAgent, WinFirstAgent, RandomAgent)) for agent in table[1:])


def test_env_runs_with_opponent_pool():
    env = MahjongSingleAgentEnv(
        {
            "opponent_agent": "pool",
            "max_steps_per_game": 120,
            "opponent_pool": {
                "members": [
                    {"kind": "heuristic", "weight": 0.7},
                    {"kind": "random", "weight": 0.3},
                ]
            },
        }
    )
    obs, info = env.reset(seed=10)
    assert obs.shape == env.observation_space.shape
    assert np.isfinite(obs).all()
    assert info["legal_actions"]
    for _ in range(5):
        obs, _, terminated, truncated, info = env.step(info["legal_actions"][0])
        assert np.isfinite(obs).all()
        if terminated or truncated:
            break
