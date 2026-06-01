<#
.SYNOPSIS
    Register (or re-register) the "revbot" scheduled task from revbot-task.xml.

.DESCRIPTION
    Imports scheduler\revbot-task.xml as a Windows scheduled task named
    "revbot" that runs under your account. Re-running it updates the task.

    Run from an ordinary (non-admin) PowerShell — the task uses LeastPrivilege
    and your interactive token, so no elevation is needed for a per-user task.

.PARAMETER RepoPath
    Path to your revbot checkout. Defaults to C:\revbot. If you cloned
    elsewhere, pass -RepoPath and the script rewrites the XML paths to match.

.EXAMPLE
    .\scheduler\install_task.ps1
.EXAMPLE
    .\scheduler\install_task.ps1 -RepoPath D:\trading\revbot
#>
[CmdletBinding()]
param(
    [string]$RepoPath = "C:\revbot",
    [string]$TaskName = "revbot"
)

$ErrorActionPreference = "Stop"
$xmlPath = Join-Path $PSScriptRoot "revbot-task.xml"
if (-not (Test-Path $xmlPath)) { throw "Cannot find $xmlPath" }

# Patch the hard-coded C:\revbot paths if the repo lives somewhere else.
$xml = Get-Content -Raw $xmlPath
if ($RepoPath -ne "C:\revbot") {
    Write-Host "Rewriting task paths to $RepoPath"
    $xml = $xml -replace 'C:\\revbot', ($RepoPath -replace '\\', '\\')
}

# Task Scheduler reparses the XML from this in-memory (UTF-16) string and
# chokes on the file's `encoding="UTF-8"` declaration ("unable to switch the
# encoding"). Strip the declaration so it parses regardless of file encoding.
$xml = $xml -replace '^\s*<\?xml.*?\?>\s*', ''

try {
    Register-ScheduledTask -TaskName $TaskName -Xml $xml -Force -ErrorAction Stop | Out-Null
} catch {
    Write-Error "Failed to register scheduled task '$TaskName': $_"
    exit 1
}

Write-Host ""
Write-Host "Registered scheduled task '$TaskName'." -ForegroundColor Green
Write-Host "  Triggers : weekdays 08:25 local time, and at logon"
Write-Host "  Runs     : powershell -File $RepoPath\run_bot.ps1"
Write-Host ""
Write-Host "Useful commands:"
Write-Host "  Start now : Start-ScheduledTask -TaskName $TaskName"
Write-Host "  Status    : Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo"
Write-Host "  Stop      : Stop-ScheduledTask  -TaskName $TaskName"
Write-Host "  Remove    : Unregister-ScheduledTask -TaskName $TaskName -Confirm:`$false"
Write-Host "  Logs      : Get-Content $RepoPath\logs\bot_<date>.log -Wait"
