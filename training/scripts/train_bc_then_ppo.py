#!/usr/bin/env python3
"""Behavior-clone the exported human traces, then fine-tune with MaskablePPO.

The BC phase maximizes log-probability of the human-chosen actions on the
same policy network that PPO will later optimize. After BC, the script runs
the normal PPO `learn` loop, so human style is injected before reinforcement
learning takes over.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3.common.callbacks import BaseCallback

from mahjong_ai.train.train_ppo import _learning_rate, build_env, build_policy_kwargs, load_config


def _load_npz(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load one heuristic-BC npz shard (observations -> static + extra keys)."""
    data = np.load(path, allow_pickle=False)
    arrays: dict[str, np.ndarray] = {}
    for key in ("observations", "table", "hand", "hand_mask", "history", "history_mask"):
        if key in data.files:
            arrays["static" if key == "observations" else key] = np.asarray(data[key], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.int64)
    return arrays, actions


def _load_jsonl(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    arrays: dict[str, list] = {}
    actions: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            trace = json.loads(line)
            actions.append(int(trace["action"]))
            for key in ("observation", "table", "hand", "hand_mask", "history", "history_mask"):
                if key in trace:
                    arrays.setdefault(key, []).append(trace[key])
    obs_arrays = {k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()}
    if "observation" in obs_arrays:
        obs_arrays["static"] = obs_arrays.pop("observation")
    return obs_arrays, np.asarray(actions, dtype=np.int64)


def load_traces(path_pattern: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load one or more BC datasets: comma-separated paths and/or glob patterns.

    Accepts heuristic npz shards (``*.npz``), human/exported jsonl traces, or a
    mix of both. All datasets must share the same observation keys.
    """
    pieces = [piece.strip() for piece in path_pattern.split(",") if piece.strip()]
    paths: list[Path] = []
    for piece in pieces:
        matches = sorted(Path(match) for match in glob.glob(str(Path(piece))))
        paths.extend(matches if matches else [Path(piece)])

    obs_parts: list[dict[str, np.ndarray]] = []
    action_parts: list[np.ndarray] = []
    for path in paths:
        obs, actions = _load_npz(path) if path.suffix == ".npz" else _load_jsonl(path)
        if len(actions) == 0:
            print(f"    (empty dataset: {path})", file=sys.stderr)
            continue
        print(f"    loaded {len(actions)} traces from {path}")
        obs_parts.append(obs)
        action_parts.append(actions)

    if not obs_parts:
        raise ValueError(f"no BC traces found for: {path_pattern}")
    obs_arrays = {key: np.concatenate([part[key] for part in obs_parts]) for key in obs_parts[0]}
    actions = np.concatenate(action_parts)
    print(f"    total {len(actions)} BC traces")
    return obs_arrays, actions


def _obs_batch(obs_arrays: dict[str, np.ndarray], idx: np.ndarray, device) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value[idx], device=device) for key, value in obs_arrays.items()}


def run_bc(model, obs_arrays: dict[str, np.ndarray], actions: np.ndarray, epochs: int, batch_size: int) -> None:
    policy = model.policy
    optimizer = policy.optimizer
    device = model.device
    total = len(actions)
    rng = np.random.default_rng(2026)
    for epoch in range(epochs):
        perm = rng.permutation(total)
        total_loss = 0.0
        batches = 0
        for start in range(0, total, batch_size):
            idx = perm[start : start + batch_size]
            obs = _obs_batch(obs_arrays, idx, device)
            act = torch.as_tensor(actions[idx], device=device)
            _, log_prob, _ = policy.evaluate_actions(obs, act)
            loss = -log_prob.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(idx)
            batches += 1
        print(f"BC epoch {epoch + 1}/{epochs} avg_loss={total_loss / max(1, total):.4f}")


class BcAuxCallback(BaseCallback):
    """Keep pulling the policy toward human actions during PPO."""

    def __init__(self, obs_arrays: dict[str, np.ndarray], actions: np.ndarray, batch_size: int, steps: int):
        super().__init__()
        self.obs_arrays = obs_arrays
        self.actions = actions
        self.batch_size = batch_size
        self.steps = steps

    def _on_step(self) -> bool:
        return True

    def on_rollout_end(self) -> None:
        policy = self.model.policy
        optimizer = policy.optimizer
        device = self.model.device
        total = len(self.actions)
        if total == 0 or self.steps <= 0:
            return
        idx = np.random.default_rng().integers(0, total, size=(self.steps, self.batch_size))
        for batch in idx:
            obs = _obs_batch(self.obs_arrays, batch, device)
            act = torch.as_tensor(self.actions[batch], device=device)
            _, log_prob, _ = policy.evaluate_actions(obs, act)
            loss = -log_prob.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--bc-data", required=True, help="JSONL produced by export_human_traces.py")
    parser.add_argument("--bc-epochs", type=int, default=8)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--output-dir", default="artifacts/checkpoints/bc_then_ppo")
    parser.add_argument("--resume", default=None, help="Skip BC and continue PPO from a saved checkpoint.")
    parser.add_argument("--bc-aux-steps", type=int, default=0, help="BC gradient steps per rollout during PPO.")
    parser.add_argument("--bc-aux-batch", type=int, default=256)
    parser.add_argument("--no-bc", action="store_true", help="Skip BC pretraining and train PPO from scratch.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = build_env(cfg)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    from sb3_contrib import MaskablePPO

    if args.resume:
        model = MaskablePPO.load(
            args.resume,
            env=env,
            device=model_cfg.get("device", "auto"),
            custom_objects={
                "learning_rate": _learning_rate(train_cfg),
                "clip_range": float(train_cfg.get("clip_range", 0.2)),
                "n_steps": int(train_cfg.get("n_steps", 128)),
                "batch_size": int(train_cfg.get("batch_size", 256)),
                "n_epochs": int(train_cfg.get("n_epochs", 4)),
                "gamma": float(train_cfg.get("gamma", 0.99)),
                "gae_lambda": float(train_cfg.get("gae_lambda", 0.95)),
                "ent_coef": float(train_cfg.get("ent_coef", 0.0)),
                "vf_coef": float(train_cfg.get("vf_coef", 0.5)),
                "max_grad_norm": float(train_cfg.get("max_grad_norm", 0.5)),
            },
        )
        model.verbose = 1
    else:
        model = MaskablePPO(
            model_cfg.get("policy", "MultiInputPolicy"),
            env,
            device=model_cfg.get("device", "auto"),
            policy_kwargs=build_policy_kwargs(model_cfg),
            learning_rate=_learning_rate(train_cfg),
            n_steps=int(train_cfg.get("n_steps", 128)),
            batch_size=int(train_cfg.get("batch_size", 256)),
            n_epochs=int(train_cfg.get("n_epochs", 4)),
            gamma=float(train_cfg.get("gamma", 0.99)),
            gae_lambda=float(train_cfg.get("gae_lambda", 0.95)),
            ent_coef=float(train_cfg.get("ent_coef", 0.0)),
            vf_coef=float(train_cfg.get("vf_coef", 0.5)),
            max_grad_norm=float(train_cfg.get("max_grad_norm", 0.5)),
            clip_range=float(train_cfg.get("clip_range", 0.2)),
            verbose=1,
        )

        if not args.no_bc:
            obs_arrays, actions = load_traces(args.bc_data)
            print(f"Loaded {len(actions)} human traces; starting BC.")
            run_bc(model, obs_arrays, actions, args.bc_epochs, args.bc_batch_size)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and not args.no_bc:
        model.save(str(out / "bc_model.zip"))
        print(f"Saved BC model to {out / 'bc_model.zip'}")

    total_timesteps = int(train_cfg.get("total_timesteps", 0))
    if total_timesteps > 0:
        callback = None
        # Periodic checkpoints so an interrupted run can be resumed via
        # --resume instead of losing all PPO progress.
        checkpoint_freq = int(cfg.get("logging", {}).get("checkpoint_freq", 0))
        if checkpoint_freq > 0:
            from stable_baselines3.common.callbacks import CheckpointCallback

            save_freq = max(1, checkpoint_freq // max(1, int(train_cfg.get("num_envs", 1))))
            periodic_dir = out / "periodic"
            periodic_dir.mkdir(parents=True, exist_ok=True)
            callback = CheckpointCallback(
                save_freq=save_freq,
                save_path=str(periodic_dir),
                name_prefix="model",
                save_replay_buffer=False,
                save_vecnormalize=False,
            )
            print(f"Periodic checkpoints every {save_freq} steps -> {periodic_dir}")
        if args.bc_aux_steps > 0:
            obs_arrays, actions = load_traces(args.bc_data)
            aux_cb = BcAuxCallback(obs_arrays, actions, args.bc_aux_batch, args.bc_aux_steps)
            callback = aux_cb if callback is None else [callback, aux_cb]
            print(f"PPO fine-tune with BC aux ({args.bc_aux_steps} steps/rollout) for {total_timesteps} timesteps.")
        else:
            print(f"Starting PPO fine-tune for {total_timesteps} timesteps.")
        model.learn(total_timesteps=total_timesteps, callback=callback)
        model.save(str(out / "final_model.zip"))
        print(f"Saved fine-tuned model to {out / 'final_model.zip'}")


if __name__ == "__main__":
    main()
