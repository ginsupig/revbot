<#
.SYNOPSIS
    Weekly walk-forward autotune for revbot.

.DESCRIPTION
    Runs `autotune_run.py <Days> --symbols <Symbols>`, which rewrites .env
    (TRADE_ALLOWLIST + median fallback params) and symbol_params.json. The live
    bot reads those only at startup, so a fresh tune on Sunday is picked up by
    Monday's 08:25 launch.

    Safe to run while the bot is idle on the weekend: the bot doesn't hold these
    files open and only re-reads them on (re)start, so a mid-week write has no
    effect until the next restart. Single-instance guarded; output is appended
    to logs\autotune_YYYYMMDD.log.

.PARAMETER Python
    Python executable. Defaults to "python". A scheduled task won't have your
    venv on PATH, so pass the venv python, e.g.
    "C:\3rev\.venv\Scripts\python.exe".

.PARAMETER Days
    Walk-forward lookback window in days (default 90 — stable; avoid short
    windows, which make the allowlist swing on noise).

.PARAMETER Symbols
    Comma-separated universe to tune. Default is the leveraged set plus the
    broad ETFs, so dropped names can re-enter if they improve.

.EXAMPLE
    .\scheduler\run_autotune.ps1 -Python "C:\3rev\.venv\Scripts\python.exe"
#>
[CmdletBinding()]
param(
    [string]$Python  = "python",
    [int]$Days       = 90,
    [string]$Symbols = "TQQQ,SOXL,TECL,NVDL,TSLL,AAPU,METU,GGLL,MSFU,AMZU,SPY,QQQ,IWM,EEM,EFA,XLI"
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot  = Split-Path -Parent $scriptDir          # parent of scheduler\
Set-Location $repoRoot

# Single-instance: never run two tunes concurrently.
$createdNew = $false
$mutex = New-Object System.Threading.Mutex($true, "Global\revbot_autotune", [ref]$createdNew)
if (-not $createdNew) {
    Write-Host "[autotune] Another tune is already running. Exiting."
    exit 0
}

$logDir = Join-Path $repoRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("autotune_{0}.log" -f (Get-Date -Format "yyyyMMdd"))

function Write-Log($m) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m
    Write-Host $line
    Add-Content -Path $log -Value $line
}

try {
    Write-Log "Autotune starting: days=$Days python='$Python' symbols=$Symbols"
    $start = Get-Date

    # python's logging writes to stderr; under ErrorActionPreference='Stop'
    # PowerShell would escalate that to a terminating error. Relax around the call.
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $Python "autotune_run.py" $Days "--symbols" $Symbols *>> $log
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prevEAP

    $secs = [int]((Get-Date) - $start).TotalSeconds
    Write-Log "Autotune finished (exit=$code) in ${secs}s. .env + symbol_params.json updated; effective on next bot restart."
    exit $code
}
finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
