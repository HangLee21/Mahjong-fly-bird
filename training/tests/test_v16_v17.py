"""Tests for v16 (public history branch) and v17 (hand<->table cross-attn, 4 layers)."""

from __future__ import annotations

import numpy as np

from mahjong_ai.env.gym_env import MahjongSingleAgentEnv
from mahjong_ai.models.feature_extractor import DefenseCrossAttentionExtractor, MahjongAttentionExtractor
from mahjong_ai.train.train_ppo import build_policy_kwargs, load_config, resolve_policy


def _env_for(cfg: dict) -> MahjongSingleAgentEnv:
    env_cfg = {**cfg.get("env", {})}
    env_cfg["reward"] = cfg.get("reward", {})
    for key in ("observation", "action_features", "opponent_pool"):
        if key in cfg:
            env_cfg[key] = cfg[key]
    return MahjongSingleAgentEnv(env_cfg)


def test_v16_history_obs_and_extractor():
    cfg = load_config("configs/ppo_mahjong_attention_v16.yaml")
    env = _env_for(cfg)
    obs, info = env.reset(seed=1)
    assert "history" in obs and "history_mask" in obs
    assert obs["static"].shape[0] == 430
    assert obs["table"].shape == (4, 213)

    kwargs = build_policy_kwargs(cfg["model"])
    assert kwargs["features_extractor_class"] is MahjongAttentionExtractor

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
    action, _ = model.predict(obs, deterministic=True, action_masks=np.asarray(info["action_mask"]))
    assert 0 <= int(action) < 128


def test_v17_cross_attention_full_layers():
    cfg = load_config("configs/ppo_mahjong_attention_v17.yaml")
    env = _env_for(cfg)
    obs, info = env.reset(seed=3)
    assert obs["static"].shape[0] == 430

    kwargs = build_policy_kwargs(cfg["model"])
    assert kwargs["features_extractor_class"] is DefenseCrossAttentionExtractor
    assert kwargs["features_extractor_kwargs"]["num_layers"] == 4  # capacity matches v12

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
    action, _ = model.predict(obs, deterministic=True, action_masks=np.asarray(info["action_mask"]))
    assert 0 <= int(action) < 128


def test_v17_extractor_forward():
    import torch

    cfg = load_config("configs/ppo_mahjong_attention_v17.yaml")
    env = _env_for(cfg)
    obs, _ = env.reset(seed=5)
    extractor = DefenseCrossAttentionExtractor(env.observation_space, features_dim=1024)
    batch = {k: torch.as_tensor(np.stack([v, v]), dtype=torch.float32) for k, v in obs.items()}
    out = extractor(batch)
    assert out.shape == (2, 1024)
    assert torch.isfinite(out).all()
