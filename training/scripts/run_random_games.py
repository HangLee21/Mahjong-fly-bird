from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong_ai.agents.random_agent import RandomAgent
from mahjong_ai.env.gym_env import MahjongSingleAgentEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-games", type=int, default=100)
    args = parser.parse_args()
    env = MahjongSingleAgentEnv({"opponent_agent": "random"})
    agent = RandomAgent(seed=2026)
    wins = 0
    for seed in range(args.num_games):
        obs, info = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.act(obs, info["legal_actions"])
            obs, _, terminated, truncated, info = env.step(action)
        wins += int(info["winner"] == 0)
    print({"num_games": args.num_games, "controlled_wins": wins})


if __name__ == "__main__":
    main()

