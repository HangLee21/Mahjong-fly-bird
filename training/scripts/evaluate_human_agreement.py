#!/usr/bin/env python3
"""Measure how well a trained model agrees with human actions on real human traces.

For each human trace (export_human_traces.py schema: observation/table/hand/
hand_mask/legal_actions/action), run the model deterministically and report:

  - top-1 agreement (model argmax == human action)
  - top-3 / top-5 contain the human action
  - per action-type agreement (discard / pong / chow / pass / win)

Usage:
  python scripts/evaluate_human_agreement.py \
      --model artifacts/checkpoints/ppo_mahjong_attention_v6/periodic/model_20000000_steps.zip \
      --traces ../artifacts/human_traces.jsonl \
      --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong_ai.env.actions import build_action_mask, decode_action  # noqa: E402


def load_traces(path: Path) -> list[dict]:
    traces: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            trace = json.loads(line)
            if "observation" in trace:
                traces.append(trace)
    return traces


def obs_for(trace: dict) -> dict[str, np.ndarray]:
    obs: dict[str, np.ndarray] = {"static": np.asarray(trace["observation"], dtype=np.float32)}
    for key in ("table", "hand", "hand_mask", "history", "history_mask"):
        if key in trace:
            obs[key] = np.asarray(trace[key], dtype=np.float32)
    return obs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    from sb3_contrib import MaskablePPO

    print(f"loading model {args.model} ...")
    model = MaskablePPO.load(args.model, device=args.device)

    traces = load_traces(Path(args.traces))
    if not traces:
        print("no traces")
        return 1
    print(f"{len(traces)} human traces")

    top1_hits = 0
    topk_hits = 0
    type_total: Counter[str] = Counter()
    type_hits: Counter[str] = Counter()
    n_legal_miss = 0
    for trace in traces:
        legal = [int(a) for a in trace.get("legal_actions") or []]
        if not legal:
            continue
        mask = build_action_mask(legal)
        action, _ = model.predict(obs_for(trace), deterministic=True, action_masks=mask)
        predicted = int(action)
        human_action = int(trace["action"])

        if predicted == human_action:
            top1_hits += 1
        # top-k: sample predicted probabilities (deterministic argmax is not enough).
        # Use the policy distribution to find where the human action ranks.
        probs = _action_probs(model, obs_for(trace), legal)
        rank = int(np.where(probs.argsort()[::-1] == human_action)[0][0]) + 1 if human_action in legal else len(legal) + 1
        if rank <= args.top_k:
            topk_hits += 1
        if human_action not in legal:
            n_legal_miss += 1

        atype = str(decode_action(human_action).type).upper()
        type_total[atype] += 1
        if predicted == human_action:
            type_hits[atype] += 1

    n = len(traces)
    print("=" * 56)
    print(f"model            : {args.model}")
    print(f"traces           : {n}")
    print(f"top-1 agreement  : {top1_hits / n * 100:.1f}%  ({top1_hits}/{n})")
    print(f"top-{args.top_k} contains human : {topk_hits / n * 100:.1f}%  ({topk_hits}/{n})")
    print(f"human action not legal: {n_legal_miss}")
    print("-" * 56)
    print("per-type top-1 agreement:")
    for atype in sorted(type_total, key=lambda t: -type_total[t]):
        print(f"  {atype:<14} {type_hits[atype]:>3}/{type_total[atype]:<4} "
              f"{type_hits[atype] / type_total[atype] * 100:5.1f}%")
    return 0


def _action_probs(model, obs: dict[str, np.ndarray], legal: list[int]) -> np.ndarray:
    """Get the model's probability over legal actions for ranking."""
    import torch

    policy = model.policy
    obs_tensor = {k: torch.as_tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in obs.items()}
    mask_tensor = torch.as_tensor(build_action_mask(legal), dtype=torch.bool).unsqueeze(0)
    with torch.no_grad():
        dist = policy.get_distribution(obs_tensor, action_masks=mask_tensor)
        probs = dist.distribution.probs.squeeze(0).cpu().numpy()
    return probs


if __name__ == "__main__":
    sys.exit(main())
