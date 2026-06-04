<#
.SYNOPSIS
    Register (or re-register) the weekly "revbot-autotune" scheduled task.

.DESCRIPTION
    Runs scheduler\run_autotune.ps1 every Sunday so Monday's bot launch starts
    on a freshly tuned allowlist + per-symbol params. A weekly 90-day tune is
    stable enough to trust while still adapting — deliberately NOT daily, which
    would churn the universe on run-to-run noise.

    Registered headless (runs whether you're logged on or not) with stored
    credentials, to match the bot task. Run from an ELEVATED PowerShell.

.PARAMETER RepoPath
    revbot checkout path (default C:\3rev).

.PARAMETER Python
    Python for the tune. Defaults to the repo's venv python.

.PARAMETER At
    Local time to run on Sunday (default 16:00 — after the trading week, so the
    90-day window includes Friday).

.EXAMPLE
    .\scheduler\install_autotune_task.ps1 -RepoPath C:\3rev
#>
[CmdletBinding()]
param(
    [string]$RepoPath = "C:\3rev",
    [string]$Python   = "",
    [string]$TaskName = "revbot-autotune",
    [string]$At       = "16:00"
)

$ErrorActionPreference = "Stop"
if (-not $Python) { $Python = Join-Path $RepoPath ".venv\Scripts\python.exe" }
$wrapper = Join-Path $RepoPath "scheduler\run_autotune.ps1"
if (-not (Test-Path $wrapper)) { throw "Cannot find $wrapper (is -RepoPath correct?)" }

$argline = '-ExecutionPolicy Bypass -NoProfile -WindowStyle Hidden -File "{0}" -Python "{1}"' -f $wrapper, $Python
$action  = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argline -WorkingDirectory $RepoPath

$trigger  = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) -RunOnlyIfNetworkAvailable

# Stored credential -> runs logged off. Get-Credential keeps the password off
# the command line / history.
$cred = Get-Credential -UserName $env:USERNAME -Message "Windows password for the headless autotune task"

try {
    Register-ScheduledTask -TaskName $TaskName -Force `
        -Action $action -Trigger $trigger -Settings $settings `
        -User $cred.UserName -Password $cred.GetNetworkCredential().Password -RunLevel Limited | Out-Null
} catch {
    Write-Error "Failed to register '$TaskName': $_"
    exit 1
}

Write-Host ""
Write-Host "Registered scheduled task '$TaskName'." -ForegroundColor Green
Write-Host "  Trigger : Sundays at $At (local), runs whether logged on or not"
Write-Host "  Runs    : powershell -File $wrapper -Python $Python"
Write-Host ""
Write-Host "  Run now : Start-ScheduledTask -TaskName $TaskName"
Write-Host "  Status  : Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo"
Write-Host "  Logs    : Get-Content $RepoPath\logs\autotune_<date>.log -Wait"
Write-Host "  Remove  : Unregister-ScheduledTask -TaskName $TaskName -Confirm:`$false"
