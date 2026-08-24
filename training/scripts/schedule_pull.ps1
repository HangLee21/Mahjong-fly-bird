# Creates (or updates) the Windows scheduled task that pulls human game data
# daily from the production backend before the 7-day cleanup can remove it.
#
# Run from an elevated shell:
#   powershell -ExecutionPolicy Bypass -File scripts\schedule_pull.ps1
param(
    [string]$TaskName = "Mahjong-FlyBird-PullHumanData",
    [string]$Time = "04:00"
)

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$bat = Join-Path $root "scripts\run_pull.bat"

$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$bat`""
$trigger = New-ScheduledTaskTrigger -Daily -At $Time
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

# Prefer SYSTEM so it runs even when no user is logged in. Falls back to the
# current user if SYSTEM cannot be used.
try {
    $principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Limited
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
        -Settings $settings -Principal $principal -Force | Out-Null
    Write-Host "Scheduled task '$TaskName' created (SYSTEM, daily $Time)."
} catch {
    Write-Warning "SYSTEM principal failed ($_); falling back to current user."
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
        -Settings $settings -Principal $principal -Force | Out-Null
    Write-Host "Scheduled task '$TaskName' created (current user, daily $Time; requires login)."
}
