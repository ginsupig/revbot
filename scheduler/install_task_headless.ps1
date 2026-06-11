<#
.SYNOPSIS
    Register the "revbot" bot task to run HEADLESS — whether you're logged on
    or not.

.DESCRIPTION
    The default installer (install_task.ps1 / register-task.ps1) imports
    revbot-task.xml, whose principal is `InteractiveToken` — so the 08:25
    trigger only fires while you are actually logged on. If the machine is at
    the lock screen, signed out, or asleep without a session, the bot never
    starts (the classic "it didn't fire this morning").

    This installer instead stores your Windows credential against the task and
    runs it at `LeastPrivilege` (RunLevel Limited) "whether the user is logged
    on or not" — the same pattern the weekly autotune task already uses. Same
    wrapper (run_bot.ps1), same lifecycle (fire weekdays 08:25 local + at logon,
    self-exit at the close, crash-restart with backoff, PT7H backstop).

    Run from an ELEVATED PowerShell (registering with a stored password needs
    it). The task itself still runs un-elevated as you.

.PARAMETER RepoPath
    revbot checkout path (default C:\revbot). Sets the working directory and the
    run_bot.ps1 path, so .env is read from here.

.PARAMETER At
    Local time to start on weekdays (default 08:25 — 5 min before the US open).

.PARAMETER TaskName
    Scheduled-task name (default "revbot").

.EXAMPLE
    # Elevated PowerShell:
    .\scheduler\install_task_headless.ps1
.EXAMPLE
    .\scheduler\install_task_headless.ps1 -RepoPath D:\trading\revbot -At 08:25
#>
[CmdletBinding()]
param(
    [string]$RepoPath = "C:\revbot",
    [string]$At       = "08:25",
    [string]$TaskName = "revbot"
)

$ErrorActionPreference = "Stop"

# Registering with a stored password requires elevation (Access denied / 0x5
# otherwise). The task still RUNS un-elevated (LeastPrivilege) as you.
$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Error "Registering a headless task with stored credentials needs elevation. Re-run this in a PowerShell opened with 'Run as administrator'."
    exit 1
}

$wrapper = Join-Path $RepoPath "run_bot.ps1"
if (-not (Test-Path $wrapper)) { throw "Cannot find $wrapper (is -RepoPath correct?)" }

$argline = '-ExecutionPolicy Bypass -NoProfile -WindowStyle Hidden -File "{0}"' -f $wrapper
$action  = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argline -WorkingDirectory $RepoPath

# Two triggers, matching revbot-task.xml: a fresh weekday start before the open,
# plus a logon trigger so a reboot mid-session brings it back (run_bot.ps1's
# mutex makes the overlap harmless; after hours main.py self-exits at once).
$daily = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $At
$logon = New-ScheduledTaskTrigger -AtLogOn

# Mirror revbot-task.xml's <Settings>: start if a trigger was missed, ignore
# overlapping launches, require network, don't let battery state block/stop it,
# wake the box for the pre-open start, force-stop ~7h after start as a backstop
# to the bot's own clean close, and add Task Scheduler's own crash-restart.
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -RunOnlyIfNetworkAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -WakeToRun `
    -ExecutionTimeLimit (New-TimeSpan -Hours 7) `
    -RestartInterval (New-TimeSpan -Minutes 1) -RestartCount 3

# Stored credential -> runs logged off. Get-Credential keeps the password off
# the command line / history. -RunLevel Limited = LeastPrivilege (un-elevated).
$cred = Get-Credential -UserName $env:USERNAME -Message "Windows password for the headless revbot task"

try {
    Register-ScheduledTask -TaskName $TaskName -Force `
        -Action $action -Trigger @($daily, $logon) -Settings $settings `
        -User $cred.UserName -Password $cred.GetNetworkCredential().Password -RunLevel Limited | Out-Null
} catch {
    Write-Error "Failed to register '$TaskName': $_"
    exit 1
}

Write-Host ""
Write-Host "Registered HEADLESS scheduled task '$TaskName'." -ForegroundColor Green
Write-Host "  Trigger : weekdays at $At (local) + at logon, runs whether logged on or not"
Write-Host "  Runs    : powershell -File $wrapper"
Write-Host ""
Write-Host "  Run now : Start-ScheduledTask -TaskName $TaskName"
Write-Host "  Status  : Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo"
Write-Host "  Logs    : Get-Content $RepoPath\logs\bot_<date>.log -Wait"
Write-Host "  Remove  : Unregister-ScheduledTask -TaskName $TaskName -Confirm:`$false"
Write-Host ""
Write-Host "Note: paper-test a few SCHEDULED runs before pointing .env at the live URL." -ForegroundColor Yellow
