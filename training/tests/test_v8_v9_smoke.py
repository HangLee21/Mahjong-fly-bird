"""Smoke tests for the v8 (heuristic BC + action features) and v9
(defense cross-attention) training paths introduced for options 2 and 3."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mahjong_ai.env.gym_env import MahjongSingleAgentEnv
from mahjong_ai.models.feature_extractor import DefenseCrossAttentionExtractor
from mahjong_ai.train.train_ppo import build_policy_kwargs, load_config, resolve_policy

ROOT = Path(__file__).resolve().parents[1]


def _env_for(cfg: dict) -> MahjongSingleAgentEnv:
    env_cfg = {**cfg.get("env", {})}
    env_cfg["reward"] = cfg.get("reward", {})
    for key in ("observation", "action_features", "opponent_pool"):
        if key in cfg:
            env_cfg[key] = cfg[key]
    return MahjongSingleAgentEnv(env_cfg)


def test_v8_config_wiring():
    cfg = load_config(str(ROOT / "configs/ppo_mahjong_attention_v8.yaml"))
    env = _env_for(cfg)
    obs, info = env.reset(seed=1)
    # v8 intentionally drops action features so heuristic shards and human
    # traces (394-dim static) can share one BC dataset.
    assert obs["static"].shape[0] == 394
    assert "table" in obs and "hand" in obs and "hand_mask" in obs
    assert len(info["legal_actions"]) > 0

    from sb3_contrib import MaskablePPO

    kwargs = build_policy_kwargs(cfg["model"])
    from mahjong_ai.models.feature_extractor import MahjongAttentionExtractor

    assert kwargs["features_extractor_class"] is MahjongAttentionExtractor
    model = MaskablePPO(
        resolve_policy(cfg["model"]),
        env,
        device="cpu",
        policy_kwargs=kwargs,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        verbose=0,
    )
    action, _ = model.predict(obs, deterministic=True, action_masks=np.asarray(info["action_mask"]))
    assert 0 <= int(action) < 128


def test_v9_defense_extractor_wiring():
    cfg = load_config(str(ROOT / "configs/ppo_mahjong_attention_v9.yaml"))
    env = _env_for(cfg)
    obs, info = env.reset(seed=3)
    assert set(("static", "hand", "hand_mask", "table")).issubset(obs.keys())
    assert obs["static"].shape[0] == 394

    kwargs = build_policy_kwargs(cfg["model"])
    assert kwargs["features_extractor_class"] is DefenseCrossAttentionExtractor

    from sb3_contrib import MaskablePPO

    model = MaskablePPO(
        resolve_policy(cfg["model"]),
        env,
        device="cpu",
        policy_kwargs=kwargs,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        verbose=0,
    )
    # Step a few real env transitions through the model.
    for _ in range(5):
        action, _ = model.predict(obs, deterministic=True, action_masks=np.asarray(info["action_mask"]))
        obs, reward, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            obs, info = env.reset(seed=4)


def test_v9_extractor_forward_shapes():
    """Direct forward pass of the defense extractor on a batch of obs."""
    cfg = load_config(str(ROOT / "configs/ppo_mahjong_attention_v9.yaml"))
    env = _env_for(cfg)
    obs, _ = env.reset(seed=5)

    import torch

    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    extractor = DefenseCrossAttentionExtractor(env.observation_space, features_dim=1024)
    batch = {k: torch.as_tensor(np.stack([v, v]), dtype=torch.float32) for k, v in obs.items()}
    out = extractor(batch)
    assert out.shape == (2, 1024)
    assert torch.isfinite(out).all()


def test_v9_fast_config_wiring():
    """v9-fast (n_epochs 2, batch = n_steps x num_envs) builds and steps."""
    cfg = load_config(str(ROOT / "configs/ppo_mahjong_attention_v9_fast.yaml"))
    env = _env_for(cfg)
    obs, info = env.reset(seed=7)
    assert obs["static"].shape[0] == 394

    kwargs = build_policy_kwargs(cfg["model"])
    assert kwargs["features_extractor_class"] is DefenseCrossAttentionExtractor

    from sb3_contrib import MaskablePPO

    model = MaskablePPO(
        resolve_policy(cfg["model"]),
        env,
        device="cpu",
        policy_kwargs=kwargs,
        n_steps=int(cfg["train"]["n_steps"]),
        batch_size=int(cfg["train"]["batch_size"]),
        n_epochs=int(cfg["train"]["n_epochs"]),
        verbose=0,
    )
    action, _ = model.predict(obs, deterministic=True, action_masks=np.asarray(info["action_mask"]))
    assert 0 <= int(action) < 128


def test_v9_fast_lr_schedule():
    """lr_schedule: linear decays 3e-5 -> 3e-6 over the run."""
    from mahjong_ai.train.train_ppo import _learning_rate

    cfg = load_config(str(ROOT / "configs/ppo_mahjong_attention_v9_fast.yaml"))
    lr = _learning_rate(cfg["train"])
    assert callable(lr)
    assert abs(lr(1.0) - 3e-5) < 1e-12
    assert abs(lr(0.5) - 1.65e-5) < 1e-12
    assert abs(lr(0.0) - 3e-6) < 1e-12
    # fixed config stays a float
    v9 = load_config(str(ROOT / "configs/ppo_mahjong_attention_v9.yaml"))
    assert _learning_rate(v9["train"]) == 3e-5
