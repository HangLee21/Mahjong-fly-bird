# Launch v11: v10 + 万筒互换数据增强 (augment_suit).
# BC warm-start on augmented heuristic shards, then PPO 20M with linear lr decay.
# From the training directory:
#   powershell -ExecutionPolicy Bypass -File scripts\launch_v11.ps1
$ErrorActionPreference = "Continue"
$log = "artifacts\train_v11.log"
& D:\MiniConda\python.exe scripts\train_bc_then_ppo.py `
    --config configs\ppo_mahjong_attention_v11.yaml `
    --bc-data "artifacts/datasets/heuristic_v11_p*_*.npz" `
    --bc-epochs 8 `
    --bc-batch-size 512 `
    --output-dir artifacts\checkpoints\ppo_mahjong_attention_v11 *>> $log
Write-Output "TRAIN_EXIT=$LASTEXITCODE" | Out-File -Append $log
