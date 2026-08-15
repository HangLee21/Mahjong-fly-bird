#!/usr/bin/env python3
"""Behavior-clone the exported human traces, then fine-tune with MaskablePPO.

The BC phase maximizes log-probability of the human-chosen actions on the
same policy network that PPO will later optimize. After BC, the script runs
the normal PPO `learn` loop, so human style is injected before reinforcement
learning takes over.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong_ai.train.train_ppo import build_env, build_policy_kwargs, load_config


def load_traces(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    static: list[list[float]] = []
    table: list[list[list[float]]] = []
    actions: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            trace = json.loads(line)
            static.append(trace["observation"])
            table.append(trace["table"])
            actions.append(int(trace["action"]))
    return (
        np.asarray(static, dtype=np.float32),
        np.asarray(table, dtype=np.float32),
        np.asarray(actions, dtype=np.int64),
    )


def run_bc(model, static: np.ndarray, table: np.ndarray, actions: np.ndarray, epochs: int, batch_size: int) -> None:
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
            obs = {
                "static": torch.as_tensor(static[idx], device=device),
                "table": torch.as_tensor(table[idx], device=device),
            }
            act = torch.as_tensor(actions[idx], device=device)
            _, log_prob, _ = policy.evaluate_actions(obs, act)
            loss = -log_prob.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(idx)
            batches += 1
        print(f"BC epoch {epoch + 1}/{epochs} avg_loss={total_loss / max(1, total):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--bc-data", required=True, help="JSONL produced by export_human_traces.py")
    parser.add_argument("--bc-epochs", type=int, default=8)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--output-dir", default="artifacts/checkpoints/bc_then_ppo")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = build_env(cfg)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    from sb3_contrib import MaskablePPO

    model = MaskablePPO(
        model_cfg.get("policy", "MultiInputPolicy"),
        env,
        device=model_cfg.get("device", "auto"),
        policy_kwargs=build_policy_kwargs(model_cfg),
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
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

    static, table, actions = load_traces(Path(args.bc_data))
    print(f"Loaded {len(actions)} human traces; starting BC.")
    run_bc(model, static, table, actions, args.bc_epochs, args.bc_batch_size)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out / "bc_model.zip"))
    print(f"Saved BC model to {out / 'bc_model.zip'}")

    total_timesteps = int(train_cfg.get("total_timesteps", 0))
    if total_timesteps > 0:
        print(f"Starting PPO fine-tune for {total_timesteps} timesteps.")
        model.learn(total_timesteps=total_timesteps)
        model.save(str(out / "final_model.zip"))
        print(f"Saved fine-tuned model to {out / 'final_model.zip'}")


if __name__ == "__main__":
    main()
