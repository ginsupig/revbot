<#
.SYNOPSIS
    Register (or refresh) the "revbot" scheduled task from revbot-task.xml.

.DESCRIPTION
    `schtasks /Create /XML` and `Register-ScheduledTask -Xml` both choke on an
    `<?xml ... encoding="UTF-8"?>` declaration (HRESULT 0x8004131a, "unable to
    switch the encoding") because the Task Scheduler COM API holds the XML as
    UTF-16. The declaration is already stripped from revbot-task.xml, and this
    script strips it again defensively, then registers the task. Registration
    needs an ELEVATED shell (0x80070005 Access denied otherwise) — the script
    checks and tells you if you need to re-run as Administrator.

    The task still RUNS as you, at LeastPrivilege (per the XML principal) — only
    the one-time registration needs admin.

.EXAMPLE
    # In an elevated PowerShell:
    C:\revbot\scheduler\register-task.ps1
#>
[CmdletBinding()]
param(
    [string]$TaskName = "revbot"
)
$ErrorActionPreference = "Stop"

# Must be elevated — registering a task otherwise fails with Access denied.
$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Error "Registering a scheduled task needs elevation. Re-run this in a PowerShell opened with 'Run as administrator'."
    exit 1
}

$here    = Split-Path -Parent $MyInvocation.MyCommand.Path
$xmlPath = Join-Path $here "revbot-task.xml"
if (-not (Test-Path $xmlPath)) {
    Write-Error "Task XML not found at $xmlPath"
    exit 1
}

# Strip any leading <?xml ...?> declaration so the COM encoding switch can't fail.
$xml = (Get-Content $xmlPath -Raw) -replace '<\?xml.*?\?>', ''

Register-ScheduledTask -TaskName $TaskName -Xml $xml -User $env:USERNAME -Force | Out-Null

Write-Host "Registered scheduled task '$TaskName'."
Write-Host "Verify with:"
Write-Host "    schtasks /Run /TN `"$TaskName`""
Write-Host "    (Get-ScheduledTaskInfo -TaskName `"$TaskName`").LastTaskResult   # 0 = ok, 267009 = currently running"
