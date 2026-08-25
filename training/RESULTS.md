# Table-Attention Model Training Results

Checkpoints are stored under `artifacts/checkpoints/` (gitignored). Evaluation
uses 100 games vs the heuristic baseline unless noted.

| Checkpoint | Steps | Setup | win_rate | avg_score | deal_in | kong_rate | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bc_then_ppo_210k/final_model.zip` | 210k | BC(10ep) + PPO heuristic | 0.17 | -0.56 | 0.10 | 0.0 | **best** |
| `bc_then_ppo_180k/final_model.zip` | 180k | BC + PPO heuristic | 0.14 | -0.81 | 0.17 | 0.0 | |
| `bc_then_ppo_150k/final_model.zip` | 150k | BC + PPO heuristic | 0.075 | -0.975 | 0.20 | 0.0 | |
| `bc_then_ppo_120k_winfirst/final_model.zip` | 120k | BC + PPO win_first | 0.167 | -0.37 | 0.10 | 0.27% | 30 games |
| `bc_then_ppo_240k/final_model.zip` | 240k | BC + PPO heuristic | 0.09 | -0.96 | 0.17 | 0.0 | noisy |
| `bc_then_ppo_240k_winboost/final_model.zip` | 240k | win-boost reward on resume | 0.12 | -0.91 | 0.17 | 0.0 | reward change on resume hurt |
| `bc_then_ppo_240k_pool/final_model.zip` | 240k | mixed opponent pool | 0.10 | -0.80 | 0.13 | 0.0 | pool curriculum hurt vs heuristic |
| `bc_then_ppo_240k_bcaux/final_model.zip` | 240k | BC aux during PPO | 0.0 | -1.72 | 0.25 | - | BC aux hurt with weak data |
| `bc_then_ppo_medium/bc_model.zip` | BC only | 10 epochs, 321 traces | 0.0 | -1.3 | 0.12 | 0.0 | human data alone too weak |
| `bc_then_ppo_330k/final_model.zip` | 330k | longer heuristic PPO | 0.093 | -0.84 | 0.147 | 0.0 | 150 games; ep_rew up but win flat |
| `bc_then_ppo_minimal/final_model.zip` | 60k | minimal sparse reward from scratch | 0.0 | -1.04 | 0.13 | 0.0 | sparse reward can't learn in 60k |

## Conclusion

- The table-attention + BC-init + PPO pipeline trains on the local GPU and
  produces legal, low-kong models.
- With only 321 human traces and foreground single-GPU training, the policy
  plateaus around 10-17% win rate vs the heuristic baseline.
- At 330k steps the shaped reward improves (`ep_rew` up) but win rate stays
  flat, indicating dense shaping is partially gamed. Switching to a minimal
  sparse reward from scratch underperforms at 60k steps. Both regimes are
  limited by the small dataset and single-GPU foreground throughput.
- Reward changes, opponent-pool curricula, and BC-aux on resume all degrade the
  best checkpoint, so reward/opponent iterations should be retrained from
  scratch rather than resumed.
- Next improvement requires more human data (finished rooms now kept 7 days)
  and/or longer continuous training.

## Baseline (heuristic vs heuristic, 200 games)

| metric | heuristic | best model (210k) |
| --- | --- | --- |
| win_rate | 0.165 | 0.123 (300 games) |
| avg_score | -0.245 | -0.767 |
| avg_steps | 13.22 | 13.03 |
| deal_in | 0.12 | 0.127 |
| kong_rate | 0.026 | 0.0 |

The new attention model reaches rough parity with the heuristic baseline while
eliminating exposed-kong bias (0% vs 2.6%) and finishing marginally faster
(13.0 vs 13.2 avg steps). Win rate is slightly lower, so it is not yet clearly
superior overall, but it directly fixes the two reported problems: kong
preference and round length.

## MahjongAttention (hand-encoder) first run — 2026-08-24

Setup: `mahjong_attention` extractor (per-tile hand Transformer + seat Transformer
+ static MLP), BC(10ep, 386 traces) + PPO 500k vs heuristic, `win_fan_scale: 0.5`,
394-dim static (no hand-goal features). Eval: 500 games vs heuristic.

| metric | heuristic base | table-attn best | **mahjong-attn v1** |
| --- | --- | --- | --- |
| win_rate | 0.165 | 0.17 | **0.08** |
| avg_score | -0.245 | -0.56 | **-0.33** |
| deal_in | 0.12 | 0.10 | **0.114** |
| kong_rate | 0.026 | 0.0 | **0.0** |
| avg_steps | 13.22 | 13.03 | **13.68** |
| illegal/fallback | - | - | **0 / 0** |
| missed_win | - | - | **0** |

Key behavior: legal play only, zero kongs, never misses a win, discard quality
76.7% best-shanten. But the policy is too conservative: claim accept rate only
26.7% while missing 74.2% of shanten-improving claims, and win rate (0.08) is
below both baselines — likely from `win_fan_scale` discouraging cheap wins plus
only 500k steps. avg_score (-0.33) is the best of any model so far. Next: drop or
lower `win_fan_scale`, add win_first curriculum, extend steps, and/or enable the
hand-goal features (obs_include_hand_goal) now that the exporter supports hand
tokens.

## MahjongAttention v2 — 2026-08-25

Setup: same extractor, BC(10ep) + PPO 1M (resumed after power loss; periodic
checkpoints added), full v3-lite reward set, `win_fan_scale: 0.1`. Eval: 500
games vs heuristic.

| metric | v1 | **v2** |
| --- | --- | --- |
| win_rate | 0.08 | **0.108** |
| avg_score | -0.33 | **-0.69** |
| deal_in | 0.114 | **0.126** |
| avg_steps | 13.68 | **13.11** |
| discard_best_shanten | 76.7% | **79.6%** |
| claim_accept_rate | 26.7% | **100%** |

v2 fixed v1's passivity but swung to the opposite extreme: claim accept rate is
100% (never passes), which opens the hand constantly and hurts hand value
(avg_score dropped to -0.69). Wins are faster (13.1 steps) but cheaper
(avg_win_points 3.7). Discard quality improved (79.6%). Root cause: claim
improvement reward (0.018) dominates pass rewards, so the policy claims on every
opportunity — the "mindless claim" failure from online feedback. v3 raises
claim_same/regression penalties, adds `claim_use_wildcard_penalty` (never eat the
live xiaoji wildcard), increases step penalty / win bonus / fan rewards for
faster, bigger wins.
