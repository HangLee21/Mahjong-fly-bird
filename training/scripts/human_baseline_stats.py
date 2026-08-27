#!/usr/bin/env python3
"""Compute human baseline stats from the raw backend dump (human_steps.jsonl).

Metrics:
  - total human steps, distinct games, steps per game
  - action-type distribution (discard / pong / chow / pass / win / kongs)
  - 胡牌手 (human WIN hands) and games with a human winner
  - 暗杠 / 明杠 / 加杠 rates
  - claim vs pass behaviour

Usage:
  python scripts/human_baseline_stats.py --input artifacts/human_steps.jsonl [--games]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

CLAIM_TYPES = {
    "PONG",
    "CHOW_LEFT",
    "CHOW_MIDDLE",
    "CHOW_RIGHT",
    "KONG_EXPOSED",
    "KONG_CONCEALED",
    "KONG_ADDED",
    "WIN",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/human_steps.jsonl")
    parser.add_argument("--games", action="store_true", help="Print per-game summaries.")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"input not found: {path}", file=sys.stderr)
        return 1
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        print("no records")
        return 1

    type_counter: Counter[str] = Counter()
    per_game: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        atype = str((record.get("action") or {}).get("type") or "").upper()
        type_counter[atype] += 1
        per_game[record.get("gameId")].append(record)

    total = len(records)
    games = len(per_game)
    steps_per_game = total / games

    wins = type_counter["WIN"]
    kong_concealed = type_counter["KONG_CONCEALED"]
    kong_exposed = type_counter["KONG_EXPOSED"]
    kong_added = type_counter["KONG_ADDED"]
    kongs = kong_concealed + kong_exposed + kong_added
    discards = type_counter["DISCARD"]
    claims = sum(type_counter[t] for t in CLAIM_TYPES)
    passes = type_counter["PASS"]
    others = {t: c for t, c in type_counter.items() if t not in CLAIM_TYPES and t != "PASS" and t != "DISCARD"}

    games_with_human_win = sum(1 for recs in per_game.values() if any(str((r.get("action") or {}).get("type") or "").upper() == "WIN" for r in recs))

    print("=" * 60)
    print("Human baseline stats")
    print("=" * 60)
    print(f"human steps         : {total}")
    print(f"games               : {games}")
    print(f"steps per game      : {steps_per_game:.2f}")
    print(f"discard rate        : {discards / total * 100:.1f}%  ({discards})")
    print(f"claim rate          : {claims / total * 100:.1f}%  ({claims})")
    print(f"pass rate           : {passes / total * 100:.1f}%  ({passes})")
    if claims + passes:
        print(f"claim/(claim+pass)  : {claims / (claims + passes) * 100:.1f}%")
    print("-" * 60)
    print(f"胡牌手 (human WIN)  : {wins}")
    print(f"games w/ human win  : {games_with_human_win} / {games} ({games_with_human_win / games * 100:.1f}%)")
    print("-" * 60)
    print(f"暗杠 (concealed)    : {kong_concealed}  ({kong_concealed / total * 100:.2f}% of steps)")
    print(f"明杠 (exposed)      : {kong_exposed}  ({kong_exposed / total * 100:.2f}% of steps)")
    print(f"加杠 (added)        : {kong_added}  ({kong_added / total * 100:.2f}% of steps)")
    if kongs:
        print(f"kong breakdown      : concealed {kong_concealed / kongs * 100:.0f}% / exposed {kong_exposed / kongs * 100:.0f}% / added {kong_added / kongs * 100:.0f}%")
    print("-" * 60)
    print("action type distribution:")
    for atype in sorted(type_counter, key=lambda t: -type_counter[t]):
        print(f"  {atype:<16} {type_counter[atype]:>4}  {type_counter[atype] / total * 100:5.1f}%")
    if others:
        print(f"  (other/unknown    : {others})")

    if args.games:
        print("-" * 60)
        print("per-game summaries:")
        for game_id in sorted(per_game):
            recs = per_game[game_id]
            gtypes = Counter(str((r.get("action") or {}).get("type") or "").upper() for r in recs)
            humans = {str((r.get("view") or {}).get("self", {}).get("userId")) for r in recs}
            print(f"  {game_id}: steps={len(recs)} players={len(humans)} "
                  f"types={dict(gtypes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
