# Mahjong Flybird Training

This folder contains a first runnable training-side implementation for Qujing
Flybird Mahjong.

The rule engine is deterministic and separate from the learning code. Agents and
models only select from legal action ids produced by the rule adapter.

Quick checks:

```bash
cd training
python -m pytest -q
python scripts/check_env.py
python scripts/run_random_games.py --num-games 20
```

GPU debug training:

```bash
python -m mahjong_ai.train.train_ppo --config configs/ppo_gpu_debug.yaml
```

Longer first production candidate training:

```bash
python -m mahjong_ai.train.train_ppo --config configs/ppo_gpu_small.yaml
```

This config saves periodic checkpoints every 100,000 timesteps and enables
early stopping after 1,000,000 timesteps if `avg_score` fails to improve by
`0.01` for 5 evaluations. The best checkpoint is saved as:

```text
artifacts/checkpoints/ppo_gpu_small/best_model.zip
```

Stronger architecture experiment:

```bash
python -m mahjong_ai.train.train_ppo --config configs/ppo_gpu_strong.yaml
```

`ppo_gpu_strong.yaml` uses a LayerNorm MLP feature extractor plus deeper
separate policy/value networks. It is slower, so compare checkpoints against
`ppo_gpu_small.yaml` by evaluation metrics instead of assuming it is better.

16GB GPU utilization experiment:

```bash
python -m mahjong_ai.train.train_ppo --config configs/ppo_gpu_16g.yaml
```

This uses 8 subprocess environments plus a larger network/batch. If Windows
process overhead is high, reduce `train.num_envs` to 4 or switch
`train.vec_env_type` to `dummy`.

Resume a small-architecture checkpoint with higher GPU/CPU utilization:

```bash
python -m mahjong_ai.train.train_ppo \
  --config configs/ppo_gpu_small_16g_resume.yaml \
  --resume artifacts/checkpoints/ppo_gpu_small/periodic/model_1300000_steps.zip
```

Use a resume config with the same model architecture as the checkpoint. You can
change `num_envs`, `batch_size`, `learning_rate`, and reward settings, but do
not switch a small checkpoint directly into the larger `ppo_gpu_16g.yaml`
architecture.

Early-stop smoke test:

```bash
python -m mahjong_ai.train.train_ppo --config configs/ppo_earlystop_smoke.yaml
```

Production-style evaluation report:

```bash
python -m mahjong_ai.eval.evaluate \
  --model artifacts/checkpoints/ppo_gpu_small/final_model.zip \
  --num-games 20000 \
  --opponent heuristic \
  --output artifacts/reports/ppo_gpu_small_eval.json \
  --replay-output artifacts/replays/ppo_gpu_small_eval.jsonl
```
