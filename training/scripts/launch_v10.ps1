# Launch v10: BC warm-start on v10 traces + human, then PPO with value features.
# Requires heuristic_v10_p*_*.npz shards (collect via collect_heuristic_traces.py
# with configs/ppo_mahjong_attention_v10.yaml). From the training directory:
#   powershell -ExecutionPolicy Bypass -File scripts\launch_v10.ps1
$ErrorActionPreference = "Continue"
$log = "artifacts\train_v10.log"
& D:\MiniConda\python.exe scripts\train_bc_then_ppo.py `
    --config configs\ppo_mahjong_attention_v10.yaml `
    --bc-data "artifacts/datasets/heuristic_v10_p*_*.npz,../artifacts/human_traces.jsonl,artifacts/human_traces.jsonl" `
    --bc-epochs 8 `
    --bc-batch-size 512 `
    --output-dir artifacts\checkpoints\ppo_mahjong_attention_v10 *>> $log
Write-Output "TRAIN_EXIT=$LASTEXITCODE" | Out-File -Append $log
