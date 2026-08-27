# Launch v8: BC warm-start on heuristic shards + human traces, then PPO.
# Run from the training directory (paths are relative to it):
#   powershell -ExecutionPolicy Bypass -File scripts\launch_v8.ps1
$ErrorActionPreference = "Continue"
$log = "artifacts\train_v8.log"
& D:\MiniConda\python.exe scripts\train_bc_then_ppo.py `
    --config configs\ppo_mahjong_attention_v8.yaml `
    --bc-data "artifacts/datasets/heuristic_v8_p*_*.npz,../artifacts/human_traces.jsonl,artifacts/human_traces.jsonl" `
    --bc-epochs 8 `
    --bc-batch-size 512 `
    --output-dir artifacts\checkpoints\ppo_mahjong_attention_v8 *>> $log
Write-Output "TRAIN_EXIT=$LASTEXITCODE" | Out-File -Append $log
