from __future__ import annotations

import random

import numpy as np

from mahjong_ai.agents.base import BaseAgent
from mahjong_ai.env.actions import (
    ACTION_KONG_ADDED,
    ACTION_KONG_CONCEALED,
    ACTION_KONG_EXPOSED,
    ACTION_PASS,
    ACTION_PONG,
    ACTION_WIN,
    is_discard,
)


class WinFirstAgent(BaseAgent):
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def act(
        self,
        observation: np.ndarray,
        legal_actions: list[int],
        info: dict | None = None,
    ) -> int:
        if ACTION_WIN in legal_actions:
            return ACTION_WIN
        discards = [a for a in legal_actions if is_discard(a)]
        if discards:
            return int(self.rng.choice(discards))
        if ACTION_PASS in legal_actions:
            return ACTION_PASS
        return int(legal_actions[0])


class HeuristicAgent(BaseAgent):
    def __init__(self, seed: int | None = None, pong_rate: float = 0.25, kong_rate: float = 0.6):
        self.rng = random.Random(seed)
        self.pong_rate = pong_rate
        self.kong_rate = kong_rate

    def act(
        self,
        observation: np.ndarray,
        legal_actions: list[int],
        info: dict | None = None,
    ) -> int:
        if ACTION_WIN in legal_actions:
            return ACTION_WIN
        kong_actions = [a for a in (ACTION_KONG_CONCEALED, ACTION_KONG_ADDED, ACTION_KONG_EXPOSED) if a in legal_actions]
        if kong_actions and self.rng.random() < self.kong_rate:
            return kong_actions[0]
        if ACTION_PONG in legal_actions and self.rng.random() < self.pong_rate:
            return ACTION_PONG
        discards = [a for a in legal_actions if is_discard(a)]
        if discards:
            return self._choose_discard(discards, info)
        if ACTION_PASS in legal_actions:
            return ACTION_PASS
        return int(legal_actions[0])

    def _choose_discard(self, discards: list[int], info: dict | None) -> int:
        hand = list((info or {}).get("hand", []))
        if not hand:
            return int(self.rng.choice(discards))
        best_action = discards[0]
        best_score = -999
        for action in discards:
            tile = action
            same = hand.count(tile)
            neighbors = sum(hand.count(t) for t in (tile - 2, tile - 1, tile + 1, tile + 2) if 0 <= t < 34)
            honor_penalty = 1 if tile >= 27 else 0
            score = honor_penalty - same * 2 - neighbors
            if score > best_score:
                best_score = score
                best_action = action
        return int(best_action)

