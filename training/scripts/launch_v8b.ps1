# Launch v8b: resume v8-10M with linear lr decay (anti-collapse), 15M more steps.
# From the training directory:
#   powershell -ExecutionPolicy Bypass -File scripts\launch_v8b.ps1
$ErrorActionPreference = "Continue"
$log = "artifacts\train_v8b.log"
& D:\MiniConda\python.exe -m mahjong_ai.train.train_ppo `
    --config configs\ppo_mahjong_attention_v8b.yaml `
    --resume artifacts\checkpoints\ppo_mahjong_attention_v8\periodic\model_10000000_steps.zip `
    --reset-timesteps *>> $log
Write-Output "TRAIN_EXIT=$LASTEXITCODE" | Out-File -Append $log
