from __future__ import annotations

import numpy as np


class BaseAgent:
    # Rule-based agents only read `info`; model agents need the observation
    # vector. The env skips building observations for agents that don't use
    # them, which is a major speed win for opponent auto-play.
    uses_observation = False

    def act(
        self,
        observation: np.ndarray,
        legal_actions: list[int],
        info: dict | None = None,
    ) -> int:
        raise NotImplementedError

