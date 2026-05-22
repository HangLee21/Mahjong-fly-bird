import numpy as np

from mahjong_ai.env.actions import ACTION_PASS, ACTION_PONG
from mahjong_ai.env.gym_env import MahjongSingleAgentEnv
from mahjong_ai.env.observation import HISTORY_EVENT_DIM, build_observation, encode_public_history
from mahjong_ai.rules.flybird import FlybirdRuleEngine


def test_public_history_events_are_recorded_for_discard_and_pass():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=60)
    state.current_player = 0
    state.hands[0] = [5, 0, 1, 2, 3, 4, 6, 9, 10, 11, 27, 28, 29, 30]
    state.hands[1] = [5, 5, 0, 1, 2, 3, 4, 6, 9, 10, 11, 27, 28]
    state = engine.step(state, 0, 5)
    assert state.public_events[-1]["type"] == "discard"
    assert ACTION_PONG in engine.get_legal_actions(state, 1)
    state = engine.step(state, 1, ACTION_PASS)
    assert state.public_events[-1]["type"] == "pass"


def test_v3_observation_is_dict_with_history():
    engine = FlybirdRuleEngine()
    state = engine.reset(seed=61)
    state = engine.step(state, 0, state.hands[0][0])
    obs = build_observation(
        engine,
        state,
        1,
        {"observation": {"version": "obs_v3_history", "history_len": 16}},
    )
    assert set(obs) == {"static", "history", "history_mask"}
    assert obs["history"].shape == (16, HISTORY_EVENT_DIM)
    assert obs["history_mask"].sum() >= 1
    assert np.isfinite(obs["static"]).all()


def test_history_env_runs_steps():
    env = MahjongSingleAgentEnv(
        {
            "opponent_agent": "heuristic",
            "max_steps_per_game": 120,
            "observation": {"version": "obs_v3_history", "history_len": 16},
        }
    )
    obs, info = env.reset(seed=62)
    assert "history" in obs
    for _ in range(3):
        obs, _, terminated, truncated, info = env.step(info["legal_actions"][0])
        assert obs["history"].shape == (16, HISTORY_EVENT_DIM)
        if terminated or truncated:
            break


def test_encode_public_history_padding():
    history, mask = encode_public_history(
        [{"type": "discard", "player": 2, "tile": 31, "step": 5, "wall": 80}],
        player_id=0,
        history_len=4,
    )
    assert history.shape == (4, HISTORY_EVENT_DIM)
    assert mask.tolist() == [0.0, 0.0, 0.0, 1.0]
