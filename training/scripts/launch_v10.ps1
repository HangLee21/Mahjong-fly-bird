# Launch v10: BC warm-start on v10 traces, then PPO with value features.
# NOTE: human traces (export_human_traces.py, 394-dim) cannot be merged into
# the 430-dim v10 BC set; heuristic 300k traces are the BC body anyway.
# Requires heuristic_v10_p*_*.npz shards. From the training directory:
#   powershell -ExecutionPolicy Bypass -File scripts\launch_v10.ps1
$ErrorActionPreference = "Continue"
$log = "artifacts\train_v10.log"
& D:\MiniConda\python.exe scripts\train_bc_then_ppo.py `
    --config configs\ppo_mahjong_attention_v10.yaml `
    --bc-data "artifacts/datasets/heuristic_v10_p*_*.npz" `
    --bc-epochs 8 `
    --bc-batch-size 512 `
    --output-dir artifacts\checkpoints\ppo_mahjong_attention_v10 *>> $log
Write-Output "TRAIN_EXIT=$LASTEXITCODE" | Out-File -Append $log
