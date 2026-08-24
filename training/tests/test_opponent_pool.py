from mahjong_ai.agents.model_agent import ModelAgent
from mahjong_ai.agents.opponent_pool import OpponentPool


def test_model_opponents_are_cached_by_default():
    pool = OpponentPool(
        {
            "members": [
                {
                    "kind": "model",
                    "weight": 1.0,
                    "model_path": "missing_model.zip",
                    "device": "cpu",
                    "deterministic": False,
                }
            ]
        }
    )

    first = pool.sample_agent()
    second = pool.sample_agent()

    assert isinstance(first, ModelAgent)
    assert first is second


def test_model_opponent_cache_can_be_disabled():
    pool = OpponentPool(
        {
            "cache_model_agents": False,
            "members": [
                {
                    "kind": "model",
                    "weight": 1.0,
                    "model_path": "missing_model.zip",
                    "device": "cpu",
                }
            ],
        }
    )

    first = pool.sample_agent()
    second = pool.sample_agent()

    assert isinstance(first, ModelAgent)
    assert isinstance(second, ModelAgent)
    assert first is not second
