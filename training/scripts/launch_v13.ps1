# Launch v13: resume v12-20M, opponent pool incl. v12 self-play, 10M steps.
# From the training directory:
#   powershell -ExecutionPolicy Bypass -File scripts\launch_v13.ps1
$ErrorActionPreference = "Continue"
$log = "artifacts\train_v13.log"
& D:\MiniConda\python.exe -m mahjong_ai.train.train_ppo `
    --config configs\ppo_mahjong_attention_v13.yaml `
    --resume artifacts\checkpoints\ppo_mahjong_attention_v12\periodic\model_20000000_steps.zip `
    --reset-timesteps *>> $log
Write-Output "TRAIN_EXIT=$LASTEXITCODE" | Out-File -Append $log
