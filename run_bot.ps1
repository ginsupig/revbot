<#
.SYNOPSIS
    Supervisor wrapper for revbot — keeps a single live instance running.

.DESCRIPTION
    Launches `python main.py` and restarts it only if it CRASHES (non-zero
    exit). main.py runs the session itself: it sleeps until the open, trades,
    liquidates at EOD, and exits cleanly (code 0) once the market has closed
    for the day. A clean exit stops this supervisor too, so the task finishes
    and tomorrow's open trigger starts a fresh run — i.e. fire-at-open,
    stop-at-close, with crash-restart preserved.

    A global mutex guarantees only ONE supervisor runs at a time, so multiple
    Task Scheduler triggers (logon + daily) can never spawn duplicate bots
    placing duplicate orders.

    Output is appended to logs\bot_YYYYMMDD.log; supervisor events go to
    logs\supervisor_YYYYMMDD.log.

.PARAMETER Python
    Python executable to use. Defaults to "python". If your scheduled task
    runs without your PATH, pass the full path, e.g.
    "C:\Users\adria\AppData\Local\Programs\Python\Python311\python.exe".

.PARAMETER MaxBackoffSeconds
    Cap on the restart backoff after rapid crashes (default 300s).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -NoProfile -File C:\revbot\run_bot.ps1
#>
[CmdletBinding()]
param(
    [string]$Python = "python",
    [int]$MaxBackoffSeconds = 300
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# --- Single-instance guard ------------------------------------------------
# If another supervisor already holds the mutex, exit quietly. This is what
# makes it safe to wire up multiple scheduled triggers.
$createdNew = $false
$mutex = New-Object System.Threading.Mutex($true, "Global\revbot_supervisor", [ref]$createdNew)
if (-not $createdNew) {
    Write-Host "[run_bot] Another supervisor instance is already running. Exiting."
    exit 0
}

# --- Logging --------------------------------------------------------------
$logDir = Join-Path $scriptDir "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Write-Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line
    $logFile = Join-Path $logDir ("supervisor_{0}.log" -f (Get-Date -Format "yyyyMMdd"))
    Add-Content -Path $logFile -Value $line
}

Write-Log "Supervisor starting in $scriptDir (python='$Python')"

$backoff = 5
try {
    while ($true) {
        $botLog = Join-Path $logDir ("bot_{0}.log" -f (Get-Date -Format "yyyyMMdd"))
        Write-Log "Launching '$Python main.py' (output -> $botLog)"
        $start = Get-Date

        # Run the bot; append all streams (stdout + stderr) to the daily log.
        # Python's logging writes INFO lines to stderr, and under
        # ErrorActionPreference='Stop' PowerShell escalates native stderr to a
        # terminating NativeCommandError that would kill the supervisor. Switch
        # to 'Continue' around the call so normal log output can't abort us.
        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        & $Python "main.py" *>> $botLog
        $code = $LASTEXITCODE
        $ErrorActionPreference = $prevEAP

        $ranSeconds = [int]((Get-Date) - $start).TotalSeconds
        Write-Log "main.py exited (code=$code) after ${ranSeconds}s"

        # Clean exit (code 0) = the bot ended the session itself (market closed
        # for the day, or a duplicate instance). Don't restart — let the task
        # finish; tomorrow's open trigger starts a fresh run. Only a crash
        # (non-zero) gets the restart-with-backoff treatment.
        if ($code -eq 0) {
            Write-Log "Clean exit — session over for the day. Supervisor stopping until next scheduled start."
            break
        }

        # Long run -> healthy, reset backoff. Instant death -> back off so we
        # don't hot-loop on a config/credential error.
        if ($ranSeconds -ge 60) {
            $backoff = 5
        } else {
            $backoff = [Math]::Min($backoff * 2, $MaxBackoffSeconds)
        }

        Write-Log "Restarting in ${backoff}s..."
        Start-Sleep -Seconds $backoff
    }
}
finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
    Write-Log "Supervisor stopped."
}
