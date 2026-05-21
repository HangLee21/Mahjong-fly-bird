from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from mahjong_ai.agents.heuristic_agent import HeuristicAgent, WinFirstAgent
from mahjong_ai.agents.random_agent import RandomAgent
from mahjong_ai.env.actions import (
    ACTION_CHOW_LEFT,
    ACTION_CHOW_MIDDLE,
    ACTION_CHOW_RIGHT,
    ACTION_KONG_ADDED,
    ACTION_KONG_CONCEALED,
    ACTION_KONG_EXPOSED,
    ACTION_PASS,
    ACTION_PONG,
    ACTION_WIN,
    is_discard,
)
from mahjong_ai.env.gym_env import MahjongSingleAgentEnv
from mahjong_ai.inference.predictor import MahjongPredictor
from mahjong_ai.rules.flybird import WILDCARD
from mahjong_ai.utils.replay import ReplayLogger


def _make_fallback_agent(kind: str):
    if kind == "random":
        return RandomAgent(seed=7)
    if kind == "win_first":
        return WinFirstAgent(seed=7)
    return HeuristicAgent(seed=7)


def evaluate(
    model_path: str | None,
    num_games: int = 100,
    *,
    opponent: str = "heuristic",
    seed_offset: int = 0,
    replay_output: str | None = None,
    include_observation: bool = False,
) -> dict:
    env = MahjongSingleAgentEnv({"opponent_agent": opponent})
    predictor = MahjongPredictor(model_path=model_path) if model_path else None
    fallback = _make_fallback_agent(opponent)
    counters: Counter[str] = Counter()
    total_score = 0.0
    score_by_seat = [0.0, 0.0, 0.0, 0.0]
    latency_ms: list[float] = []
    action_counts: Counter[str] = Counter()
    xiaoji_discards = 0
    discard_count = 0
    replay = (
        ReplayLogger(
            replay_output,
            include_observation=include_observation,
            model_version=Path(model_path).stem if model_path else "heuristic_eval",
        )
        if replay_output
        else None
    )
    try:
        for game_index in range(num_games):
            game_id = f"eval_{seed_offset + game_index:08d}"
            obs, info = env.reset(seed=seed_offset + game_index)
            terminated = truncated = False
            game_steps = 0
            while not (terminated or truncated):
                state_hash_before = info["state_hash"]
                legal_actions = list(info["legal_actions"])
                start = time.perf_counter()
                fallback_used = False
                if predictor:
                    result = predictor.predict(obs, legal_actions)
                    action = int(result["action"])
                    fallback_used = bool(result["fallback_used"])
                else:
                    action = fallback.act(obs, legal_actions, {"hand": env.state.hands[0]})
                latency_ms.append((time.perf_counter() - start) * 1000.0)
                counters["fallback_count"] += int(fallback_used)
                _count_action(action, action_counts)
                if is_discard(action):
                    discard_count += 1
                    xiaoji_discards += int(action == WILDCARD)
                next_obs, reward, terminated, truncated, next_info = env.step(action)
                counters["illegal_action_count"] += int("illegal_action" in next_info)
                game_steps += 1
                if replay:
                    replay.log_step(
                        game_id=game_id,
                        step=game_steps,
                        player_id=0,
                        state_hash_before=state_hash_before,
                        legal_actions=legal_actions,
                        action=action,
                        action_source="model" if predictor else "heuristic",
                        state_hash_after=next_info["state_hash"],
                        reward=reward,
                        observation=obs,
                        extra={"fallback_used": fallback_used},
                    )
                obs, info = next_obs, next_info

            final_scores = list(info["scores"])
            winner = info["winner"]
            draw = bool(info["draw"])
            total_score += final_scores[0]
            for seat, score in enumerate(final_scores):
                score_by_seat[seat] += score
            counters["wins"] += int(winner == 0)
            counters["draws"] += int(draw)
            counters["controlled_deal_in"] += int(winner is not None and info["payer"] == 0)
            counters["controlled_self_draw_win"] += int(winner == 0 and info["win_type"] == "self_draw")
            counters["controlled_ron_win"] += int(winner == 0 and info["win_type"] == "ron")
            counters["truncated_games"] += int(truncated)
            counters["total_steps"] += game_steps
            if replay:
                replay.log_final(
                    game_id=game_id,
                    final_scores=final_scores,
                    winner=winner,
                    draw=draw,
                    total_steps=game_steps,
                    model_versions={"0": Path(model_path).stem if model_path else "heuristic_eval"},
                    extra={
                        "win_type": info["win_type"],
                        "payer": info["payer"],
                        "win_points": info["win_points"],
                        "win_names": info["win_names"],
                    },
                )
    finally:
        if replay:
            replay.close()

    total_wins = max(1, counters["wins"])
    return {
        "model": model_path,
        "opponent": opponent,
        "num_games": num_games,
        "avg_score": total_score / num_games,
        "win_rate": counters["wins"] / num_games,
        "deal_in_rate": counters["controlled_deal_in"] / num_games,
        "draw_rate": counters["draws"] / num_games,
        "ron_rate": counters["controlled_ron_win"] / total_wins,
        "self_draw_rate": counters["controlled_self_draw_win"] / total_wins,
        "illegal_action_count": counters["illegal_action_count"],
        "fallback_count": counters["fallback_count"],
        "truncated_games": counters["truncated_games"],
        "avg_steps": counters["total_steps"] / num_games,
        "seat_avg_score": [score / num_games for score in score_by_seat],
        "action_rates": {
            "discard": action_counts["discard"] / max(1, counters["total_steps"]),
            "pong": action_counts["pong"] / max(1, counters["total_steps"]),
            "chow": action_counts["chow"] / max(1, counters["total_steps"]),
            "kong": action_counts["kong"] / max(1, counters["total_steps"]),
            "win": action_counts["win"] / max(1, counters["total_steps"]),
            "pass": action_counts["pass"] / max(1, counters["total_steps"]),
        },
        "xiaoji_discard_rate": xiaoji_discards / max(1, discard_count),
        "model_latency_ms": {
            "mean": sum(latency_ms) / max(1, len(latency_ms)),
            "p95": _percentile(latency_ms, 95),
        },
        "replay_output": replay_output,
    }


def _count_action(action: int, counts: Counter[str]) -> None:
    if is_discard(action):
        counts["discard"] += 1
    elif action == ACTION_PONG:
        counts["pong"] += 1
    elif action in (ACTION_CHOW_LEFT, ACTION_CHOW_MIDDLE, ACTION_CHOW_RIGHT):
        counts["chow"] += 1
    elif action in (ACTION_KONG_CONCEALED, ACTION_KONG_EXPOSED, ACTION_KONG_ADDED):
        counts["kong"] += 1
    elif action == ACTION_WIN:
        counts["win"] += 1
    elif action == ACTION_PASS:
        counts["pass"] += 1


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--opponent", choices=["heuristic", "random", "win_first"], default="heuristic")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--replay-output", default=None)
    parser.add_argument("--include-observation", action="store_true")
    args = parser.parse_args()
    result = evaluate(
        args.model,
        args.num_games,
        opponent=args.opponent,
        seed_offset=args.seed_offset,
        replay_output=args.replay_output,
        include_observation=args.include_observation,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
