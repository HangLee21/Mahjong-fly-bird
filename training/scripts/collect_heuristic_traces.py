#!/usr/bin/env python3
"""Collect heuristic-expert traces as JSONL for the attention policy (v8 BC).

Each line mirrors the schema produced by export_human_traces.py so the same BC
pipeline (train_bc_then_ppo.py --bc-data) can consume it:

    {
      "observation": [...static...],
      "table": [[...] x4 seats],
      "hand": [[...] x14 tiles],
      "hand_mask": [...],
      "legal_actions": [...],
      "action": int,
      "meta": {"game_id", "step", "player", "type", "round"}
    }

The observation config (include_table / include_hand / include_action_features)
comes from the training config, so the exported features exactly match what the
attention policy consumes at inference.

Usage:
  python scripts/collect_heuristic_traces.py --config configs/ppo_mahjong_attention_v8.yaml \
      --num-samples 200000 --output artifacts/datasets/heuristic_v8_traces.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong_ai.agents.heuristic_agent import HeuristicAgent  # noqa: E402
from mahjong_ai.env.actions import decode_action  # noqa: E402
from mahjong_ai.env.gym_env import MahjongSingleAgentEnv  # noqa: E402


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_env_config(cfg: dict) -> dict:
    env_cfg = {**cfg.get("env", {})}
    env_cfg["reward"] = cfg.get("reward", {})
    for key in ("observation", "action_features", "opponent_pool"):
        if key in cfg:
            env_cfg[key] = cfg[key]
    return env_cfg


def _obs_dict(obs: object) -> dict:
    if isinstance(obs, dict):
        return obs
    return {"static": obs}


def collect_traces(
    config: dict,
    *,
    num_samples: int,
    output: str,
    seed: int | None = None,
) -> dict:
    seed = int(seed if seed is not None else config.get("seed", 2026))
    expert = HeuristicAgent(seed=seed)
    env = MahjongSingleAgentEnv(build_env_config(config))
    controlled_player = int(config.get("env", {}).get("controlled_player", 0))

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    game_count = 0
    max_games = max(1000, num_samples * 4)
    with out_path.open("w", encoding="utf-8") as out:
        while written < num_samples and game_count < max_games:
            obs, info = env.reset(seed=seed + game_count)
            terminated = truncated = False
            step = 0
            while not (terminated or truncated) and written < num_samples:
                legal_actions = list(info["legal_actions"])
                if not legal_actions:
                    break
                expert_info = {**info}
                if env.state is not None:
                    expert_info["hand"] = list(env.state.hands[controlled_player])
                action = int(expert.act(obs, legal_actions, expert_info))
                if action not in legal_actions:
                    action = int(legal_actions[0])

                obs_dict = _obs_dict(obs)
                trace = {
                    "observation": np.asarray(obs_dict["static"], dtype=np.float32).round(6).tolist(),
                    "legal_actions": legal_actions,
                    "action": action,
                    "meta": {
                        "game_id": f"heuristic-{seed}-{game_count}",
                        "step": step,
                        "player": controlled_player,
                        "type": str(decode_action(action).type).upper(),
                        "round": 0,
                    },
                }
                for key in ("table", "hand", "hand_mask", "history", "history_mask"):
                    if key in obs_dict:
                        value = np.asarray(obs_dict[key], dtype=np.float32)
                        trace[key] = value.round(6).tolist()
                out.write(json.dumps(trace, ensure_ascii=False) + "\n")
                written += 1
                step += 1

                obs, reward, terminated, truncated, info = env.step(action)
            game_count += 1
            if game_count % 50 == 0:
                print(f"    {written} traces / {game_count} games", flush=True)

    return {"output": str(out_path), "num_samples": written, "games": game_count}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-samples", type=int, default=100000)
    parser.add_argument("--output", default="artifacts/datasets/heuristic_v8_traces.jsonl")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    result = collect_traces(
        load_config(args.config),
        num_samples=args.num_samples,
        output=args.output,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
