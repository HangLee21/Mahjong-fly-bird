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
    # Action features enabled -> static dim includes the 128x18 table.
    assert obs["static"].shape[0] == 394 + 128 * 18
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
