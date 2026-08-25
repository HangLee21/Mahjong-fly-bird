"""Diagnose the child-process crash under Windows spawn."""
from __future__ import annotations

import multiprocessing as mp
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def child_entry(env_cfg: dict, rank: int, pipe) -> None:
    try:
        from mahjong_ai.env.gym_env import MahjongSingleAgentEnv

        env = MahjongSingleAgentEnv({**env_cfg, "seed_offset": 2026 + rank})
        obs, info = env.reset()
        pipe.send(("OK", list(obs.keys()) if isinstance(obs, dict) else type(obs).__name__))
    except Exception:
        pipe.send(("ERROR", traceback.format_exc()))


def main() -> None:
    env_cfg = {
        "controlled_player": 0,
        "max_steps_per_game": 120,
        "opponent_agent": "heuristic",
        "allow_chow": True,
        "observation": {"include_hand": True, "include_table": True},
        "reward": {"step_penalty": 0.003},
    }
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=child_entry, args=(env_cfg, 0, child))
    proc.start()
    child.close()
    if parent.poll(60):
        status, payload = parent.recv()
        print("child status:", status)
        if status == "OK":
            print("child env OK, obs keys:", payload)
        else:
            print(payload)
    else:
        print("TIMEOUT: child hung (no response in 60s)")
        proc.terminate()
    proc.join(timeout=5)


if __name__ == "__main__":
    main()
