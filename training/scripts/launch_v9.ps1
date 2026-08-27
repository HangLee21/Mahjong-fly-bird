# Launch v9-fast: defense cross-attention, speed-optimized PPO (option 3).
# n_epochs 2 + batch 10240 => learn 时间减半，fps 预计 ~820 -> ~1700+。
# Run after v8's GPU slot frees up. From the training directory:
#   powershell -ExecutionPolicy Bypass -File scripts\launch_v9.ps1
# 若需与 v6/v9 严格对照（n_epochs=4），改用：
#   python -m mahjong_ai.train.train_ppo --config configs/ppo_mahjong_attention_v9.yaml
$ErrorActionPreference = "Continue"
$log = "artifacts\train_v9_fast.log"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
& D:\MiniConda\python.exe -m mahjong_ai.train.train_ppo `
    --config configs\ppo_mahjong_attention_v9_fast.yaml *>> $log
Write-Output "TRAIN_EXIT=$LASTEXITCODE" | Out-File -Append $log
