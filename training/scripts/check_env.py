from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong_ai.env.gym_env import MahjongSingleAgentEnv


def main() -> None:
    env = MahjongSingleAgentEnv({"opponent_agent": "heuristic"})
    obs, info = env.reset(seed=2026)
    assert obs.shape == env.observation_space.shape
    assert info["legal_actions"]
    mask = env.action_masks()
    assert mask.shape == (env.action_space.n,)
    print("env ok", {"obs_dim": obs.shape[0], "legal_actions": info["legal_actions"][:8]})


if __name__ == "__main__":
    main()

