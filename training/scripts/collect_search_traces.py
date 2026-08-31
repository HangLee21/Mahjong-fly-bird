"""Search-enhanced behavior cloning dataset collection (v14).

Plays heuristic games; at every controlled-player decision the shallow
expectation search (mahjong_ai/search/expectation.py) scores the legal actions
and the SEARCH TEACHER's choice is recorded as the action label instead of the
heuristic's own move. The resulting traces teach the policy the search-aided
decision ("slow thinking teacher, fast thinking student") with zero search cost
during training.

Output is the same npz shard format as collect_heuristic_traces.py so the
existing BC loader (train_bc_then_ppo.py --bc-data) works unchanged.

Usage:
  python scripts/collect_search_traces.py \
      --config configs/ppo_mahjong_attention_v14.yaml \
      --num-samples 300000 --output artifacts/datasets/search_v14_traces.npz
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
from mahjong_ai.env.gym_env import MahjongSingleAgentEnv  # noqa: E402
from mahjong_ai.search.expectation import search_action_values  # noqa: E402


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


def _flush_npz(out_prefix: Path, shard_index: int, buckets: dict[str, list], actions: list[int]) -> None:
    shard = Path(f"{out_prefix}_{shard_index:05d}.npz")
    payload: dict[str, np.ndarray] = {
        "observations": np.asarray(buckets["static"], dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
    }
    for key in ("table", "hand", "hand_mask", "history", "history_mask"):
        if key in buckets and buckets[key]:
            payload[key] = np.asarray(buckets[key], dtype=np.float32)
    np.savez_compressed(shard, **payload)
    print(f"    wrote {shard} ({payload['observations'].shape[0]} samples)", flush=True)


def collect_traces(
    config: dict,
    *,
    num_samples: int,
    output: str,
    seed: int | None = None,
    n_samples: int = 4,
    depth: int = 2,
    shard_size: int = 50000,
) -> dict:
    seed = int(seed if seed is not None else config.get("seed", 2026))
    expert = HeuristicAgent(seed=seed)
    env = MahjongSingleAgentEnv(build_env_config(config))
    controlled_player = int(config.get("env", {}).get("controlled_player", 0))

    out_prefix = Path(output).with_suffix("")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list] = {}
    actions: list[int] = []
    shard_index = 0

    written = 0
    game_count = 0
    max_games = max(1000, num_samples * 4)
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
            # Search teacher decides, not the heuristic.
            values = search_action_values(
                env.state,
                controlled_player,
                legal_actions,
                n_samples=n_samples,
                depth=depth,
                seed=seed + game_count * 1000 + step,
            )
            action = max(values, key=values.get)
            if action not in legal_actions:
                action = int(legal_actions[0])

            obs_dict = obs if isinstance(obs, dict) else {"static": obs}
            for key, value in obs_dict.items():
                buckets.setdefault(key, []).append(np.asarray(value, dtype=np.float32))
            actions.append(action)

            written += 1
            step += 1
            if written % shard_size == 0:
                _flush_npz(out_prefix, shard_index, buckets, actions)
                buckets = {}
                actions = []
                shard_index += 1

            obs, reward, terminated, truncated, info = env.step(action)
        game_count += 1
        if game_count % 25 == 0:
            print(f"    {written} traces / {game_count} games", flush=True)

    if actions:
        _flush_npz(out_prefix, shard_index, buckets, actions)

    return {"output": str(output), "num_samples": written, "games": game_count}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-samples", type=int, default=300000)
    parser.add_argument("--output", default="artifacts/datasets/search_v14_traces.npz")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--shard-size", type=int, default=50000)
    args = parser.parse_args()

    result = collect_traces(
        load_config(args.config),
        num_samples=args.num_samples,
        output=args.output,
        seed=args.seed,
        n_samples=args.n_samples,
        depth=args.depth,
        shard_size=args.shard_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
