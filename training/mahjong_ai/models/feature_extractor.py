from __future__ import annotations

from typing import Any

import torch as th
from torch import nn

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except Exception:  # pragma: no cover
    BaseFeaturesExtractor = object


class LayerNormMLPExtractor(BaseFeaturesExtractor):
    """A compact feature extractor for mixed Mahjong vector observations.

    The observation contains counts, one-hot fields and scalar round state. A
    few normalized dense layers usually train more smoothly than sending the raw
    vector directly into a policy/value MLP.
    """

    def __init__(
        self,
        observation_space: Any,
        features_dim: int = 512,
        hidden_dims: list[int] | tuple[int, ...] = (512, 512),
        dropout: float = 0.0,
    ):
        super().__init__(observation_space, features_dim)
        input_dim = int(observation_space.shape[0])
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            layers.append(nn.LayerNorm(int(hidden_dim)))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, features_dim))
        layers.append(nn.LayerNorm(features_dim))
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations.float())

