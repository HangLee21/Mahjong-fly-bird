#!/usr/bin/env python3
"""Convert the DB human-step dump into training observation/action traces.

Input is the JSONL produced by `dump_human_steps.sql`. Output is a JSONL where
each line has:

    {
      "observation": [...],   # static observation vector (float32)
      "legal_actions": [...], # legal action ids before the action
      "action": int,          # the human-chosen action id
      "meta": {...}
    }

The exporter reconstructs the state *before* the human acted by inverting the
human's own action (their hand and melds/discards) and using the previous
step's last discard for discard turns.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mahjong_ai.env.observation import HAND_MAX_TILES, HAND_TOKEN_DIM, encode_hand_tokens  # noqa: E402


N_TILE_TYPES = 34
SEAT_DIM = N_TILE_TYPES + N_TILE_TYPES + 1 + 1 + 1 + 4 + 1
OBS_DIM = (
    N_TILE_TYPES  # own hand
    + N_TILE_TYPES  # own meld tiles
    + 4 * N_TILE_TYPES  # discards, per seat
    + 4 * N_TILE_TYPES  # open meld tiles, per seat
    + 4  # scores / 100
    + 4  # dealer one-hot
    + 4  # current-player one-hot
    + 4  # relative current-player one-hot
    + (N_TILE_TYPES + 1)  # last discard one-hot
    + 3  # wall / kong pool / xiaoji-disabled
)

CHOW_OFFSETS = {
    "CHOW_LEFT": (1, 2),
    "CHOW_MIDDLE": (-1, 1),
    "CHOW_RIGHT": (-2, -1),
}


def count_vec(tiles: list[int], denom: float = 4.0) -> np.ndarray:
    vec = np.zeros(N_TILE_TYPES, dtype=np.float32)
    for tile in tiles:
        if 0 <= int(tile) < N_TILE_TYPES:
            vec[int(tile)] += 1.0
    return vec / denom


def one_hot(index: int | None, size: int) -> np.ndarray:
    vec = np.zeros(size, dtype=np.float32)
    if index is not None and 0 <= int(index) < size:
        vec[int(index)] = 1.0
    return vec


def table_tokens(
    discards_before: list[list[int]],
    melds_before: list[list[dict[str, Any]]],
    scores: list[float],
    dealer: int,
    current: int,
    player_index: int,
    hand_counts: list[int],
) -> list[list[float]]:
    tokens = np.zeros((4, SEAT_DIM), dtype=np.float32)
    for seat in range(4):
        discards = count_vec(discards_before[seat])
        meld_tiles = count_vec([int(t) for m in melds_before[seat] for t in (m.get("tiles") or [])])
        score = float(scores[seat]) / 100.0
        relative = np.zeros(4, dtype=np.float32)
        relative[(seat - player_index) % 4] = 1.0
        hand_count = min(1.0, float(hand_counts[seat]) / 14.0)
        tokens[seat] = np.concatenate(
            [
                discards,
                meld_tiles,
                np.asarray([score], dtype=np.float32),
                np.asarray([1.0 if seat == dealer else 0.0], dtype=np.float32),
                np.asarray([1.0 if seat == current else 0.0], dtype=np.float32),
                relative,
                np.asarray([hand_count], dtype=np.float32),
            ]
        )
    return tokens.round(6).tolist()


def seats_from_view(view: dict[str, Any]) -> list[dict[str, Any]]:
    players = view.get("players")
    if isinstance(players, list) and players:
        by_seat = {int(p.get("seatIndex", 0)): p for p in players}
        return [by_seat.get(i, {}) for i in range(4)]
    return [{} for _ in range(4)]


def action_type(action: dict[str, Any]) -> str:
    return str(action.get("type") or "").upper()


def reconstruct_before(record: dict[str, Any]) -> dict[str, Any] | None:
    view = record.get("view") or {}
    action = record.get("action") or {}
    atype = action_type(action)
    tile = action.get("tile")
    player_index = int(record.get("playerIndex", 0))

    if atype in {
        "WIN",
        "KONG_EXPOSED",
        "KONG_CONCEALED",
        "KONG_ADDED",
        # Backend-only intermediate step where the player picks which tile to
        # kong; the training env treats kong as a single action, so there is no
        # matching training action id (109) or state transition.
        "SELECT_KONG_TILE",
    }:
        # Terminal or complex kong inversion; not included in the BC set yet.
        return None

    seats = seats_from_view(view)
    self_view = view.get("self") or {}
    hand_after = [int(t) for t in (self_view.get("hand") or [])]
    melds_after = list(self_view.get("melds") or [])
    discards = [[int(t) for t in (seats[i].get("discards") or [])] for i in range(4)]
    melds = [list(seats[i].get("melds") or []) for i in range(4)]

    hand_before = list(hand_after)
    self_melds_before = list(melds_after)
    discards_before = [list(d) for d in discards]

    if atype == "DISCARD" and tile is not None:
        hand_before.append(int(tile))
        if discards_before[player_index]:
            discards_before[player_index] = discards_before[player_index][:-1]
        last_discard = record.get("prevLastDiscard")
    elif atype == "PONG" and tile is not None:
        hand_before.extend([int(tile), int(tile)])
        if self_melds_before:
            self_melds_before = self_melds_before[:-1]
        last_discard = view.get("lastDiscard")
    elif atype in CHOW_OFFSETS and tile is not None:
        d0, d1 = CHOW_OFFSETS[atype]
        hand_before.extend([int(tile) + d0, int(tile) + d1])
        if self_melds_before:
            self_melds_before = self_melds_before[:-1]
        last_discard = view.get("lastDiscard")
    else:  # PASS
        last_discard = view.get("lastDiscard")

    melds_before = [list(m) for m in melds]
    melds_before[player_index] = list(self_melds_before)

    hand_counts = count_vec(hand_before)
    self_meld_counts = count_vec([int(t) for m in self_melds_before for t in (m.get("tiles") or [])])
    discard_counts = np.concatenate([count_vec(d) for d in discards_before]).astype(np.float32)
    open_meld_counts = np.concatenate(
        [count_vec([int(t) for m in ms for t in (m.get("tiles") or [])]) for ms in melds_before]
    ).astype(np.float32)

    scores = [float(x) for x in (view.get("scores") or [0, 0, 0, 0])]
    dealer = int(view.get("dealer", 0))
    current = int(view.get("currentPlayer", player_index))
    relative = (current - player_index) % 4

    last = np.zeros(N_TILE_TYPES + 1, dtype=np.float32)
    ld_tile = last_discard.get("tile") if isinstance(last_discard, dict) else None
    if ld_tile is None:
        last[N_TILE_TYPES] = 1.0
    else:
        last[int(ld_tile)] = 1.0

    wall = int(view.get("wallTilesRemaining", view.get("wallCount", 0)))
    kong_pool = view.get("publicKongTiles") or []
    xiaoji_disabled = 0.0 if view.get("xiaoJiActiveAsWild", True) else 1.0
    round_info = np.asarray(
        [wall / 136.0, len(kong_pool) / 2.0, xiaoji_disabled],
        dtype=np.float32,
    )

    observation = np.concatenate(
        [
            hand_counts,
            self_meld_counts,
            discard_counts,
            open_meld_counts,
            np.asarray(scores, dtype=np.float32) / 100.0,
            one_hot(dealer, 4),
            one_hot(current, 4),
            one_hot(relative, 4),
            last,
            round_info,
        ]
    ).astype(np.float32)

    if observation.shape != (OBS_DIM,):
        raise ValueError(f"observation shape {observation.shape} != ({OBS_DIM},)")

    hand_tokens, hand_mask = encode_hand_tokens(hand_before)

    return {
        "observation": observation,
        "table": table_tokens(
            discards_before,
            melds_before,
            scores,
            dealer,
            current,
            player_index,
            [
                len(hand_before) if i == player_index else int(seats[i].get("handCount", 0))
                for i in range(4)
            ],
        ),
        "hand": hand_tokens.round(6).tolist(),
        "hand_mask": hand_mask.tolist(),
        "legal_actions": [int(a) for a in (record.get("legalActions") or [])],
        "action": int(action.get("actionId", -1)),
        "meta": {
            "game_id": record.get("gameId"),
            "step": int(record.get("stepIndex", 0)),
            "player": player_index,
            "type": atype,
            "round": int(view.get("currentRound", view.get("roundIndex", 0))),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="JSONL from dump_human_steps.sql (default: stdin)")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    args = parser.parse_args()

    if args.input is not None:
        lines = args.input.read_text(encoding="utf-8").splitlines()
    else:
        lines = sys.stdin.read().splitlines()

    written = 0
    skipped = 0
    with args.output.open("w", encoding="utf-8") as out:
        for raw in lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                skipped += 1
                continue
            trace = reconstruct_before(record)
            if trace is None:
                skipped += 1
                continue
            trace["observation"] = np.asarray(trace["observation"], dtype=np.float32).round(6).tolist()
            out.write(json.dumps(trace, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote {written} traces, skipped {skipped}", file=sys.stderr)


if __name__ == "__main__":
    main()
