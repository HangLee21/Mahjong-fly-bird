# Launch v12: river-sequence + sorted hands + value features, BC + PPO.
# From the training directory:
#   powershell -ExecutionPolicy Bypass -File scripts\launch_v12.ps1
$ErrorActionPreference = "Continue"
$log = "artifacts\train_v12.log"
& D:\MiniConda\python.exe scripts\train_bc_then_ppo.py `
    --config configs\ppo_mahjong_attention_v12.yaml `
    --bc-data "artifacts/datasets/heuristic_v12_p*_*.npz" `
    --bc-epochs 8 `
    --bc-batch-size 512 `
    --output-dir artifacts\checkpoints\ppo_mahjong_attention_v12 *>> $log
Write-Output "TRAIN_EXIT=$LASTEXITCODE" | Out-File -Append $log
