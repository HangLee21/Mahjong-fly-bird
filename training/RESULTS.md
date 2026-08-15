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

## Conclusion

- The table-attention + BC-init + PPO pipeline trains on the local GPU and
  produces legal, low-kong models.
- With only 321 human traces and foreground single-GPU training, the policy
  plateaus around 10-17% win rate vs the heuristic baseline.
- Reward changes, opponent-pool curricula, and BC-aux on resume all degrade the
  best checkpoint, so reward/opponent iterations should be retrained from
  scratch rather than resumed.
- Next improvement requires more human data (finished rooms now kept 7 days)
  and/or longer continuous training.
