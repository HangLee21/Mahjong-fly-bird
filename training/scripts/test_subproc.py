"""Diagnose Windows SubprocVecEnv for the mahjong env.

Run:  python scripts/test_subproc.py
Must be a script file (not -c): Windows spawn re-imports the main module.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _make_factory(env_cfg: dict, rank: int, base_seed: int):
    def _init():
        from mahjong_ai.env.gym_env import MahjongSingleAgentEnv

        return MahjongSingleAgentEnv({**env_cfg, "seed_offset": base_seed + rank})

    return _init


def main() -> None:
    from stable_baselines3.common.vec_env import SubprocVecEnv

    env_cfg = {
        "controlled_player": 0,
        "max_steps_per_game": 120,
        "opponent_agent": "heuristic",
        "allow_chow": True,
        "observation": {"include_hand": True, "include_table": True},
        "reward": {"step_penalty": 0.003},
    }
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    print(f"creating SubprocVecEnv with {n} envs ...")
    env = SubprocVecEnv([_make_factory(env_cfg, i, 2026) for i in range(n)])
    print("created OK; observation space:", env.observation_space)
    obs = env.reset()
    print("reset OK; obs keys:", list(obs.keys()))
    t0 = time.perf_counter()
    steps = 0
    infos = [{} for _ in range(env.num_envs)]
    for i in range(100):
        actions = np.array(
            [
                infos[e].get("legal_actions", [100])[0] if infos[e].get("legal_actions") else 100
                for e in range(env.num_envs)
            ],
            dtype=np.int64,
        )
        obs, rew, dones, infos = env.step(actions)
        steps += 1
    dt = time.perf_counter() - t0
    print(f"vectorized steps OK: {steps} rollouts in {dt:.2f}s -> {n * steps / dt:.1f} env-steps/s")
    env.close()
    print("SUBPROC_OK")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
