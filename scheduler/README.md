# Scheduling revbot (Windows Task Scheduler + restart-on-failure)

These files make revbot **fire at the open and stop at the close**, every
trading day, without running it by hand.

## Lifecycle

The task starts the supervisor **weekdays at 08:25 CT** (5 min before the open).
`main.py` then runs the whole session itself: it sleeps until the bell, trades,
liquidates at EOD, and — once the market has **closed for the day** — flattens
any stragglers and **exits cleanly (code 0)**. The supervisor treats a clean
exit as "done for the day" and stops, so the task finishes. Tomorrow's 08:25
trigger starts a fresh run. A **crash** (non-zero exit) is restarted with
backoff; a clean close is not. A `PT7H` execution-time-limit backstops the
self-exit (force-stop ~15:25 CT) in case the clock check ever fails.

## Pieces

| File | Role |
|------|------|
| `../run_bot.ps1` | Supervisor. Launches `python main.py`, **restarts it on crash** with backoff, and **stops on a clean (code 0) exit** so it shuts down at the close. A global mutex ensures only **one** instance runs, so duplicate triggers can't place duplicate orders. Logs to `../logs/`. |
| `revbot-task.xml` | Scheduled-task definition: starts the supervisor **weekdays at 08:25 CT** (before US open) and **at logon** (a reboot mid-session brings it back; after hours it self-exits at once). `PT7H` time-limit backstop. |
| `install_task.ps1` | Registers/updates the task named `revbot` **for a logged-on session** (`InteractiveToken` — only fires while you're signed in). |
| `install_task_headless.ps1` | Same task, but **runs whether you're logged on or not** (stored credential, like the autotune task). Use this if the bot must start unattended. Run elevated. |
| `run_autotune.ps1` | Weekly tuner wrapper. Runs `autotune_run.py 90 --symbols …`, which rewrites `.env` (`TRADE_ALLOWLIST` + fallback params) and `symbol_params.json`. Single-instance guarded; logs to `../logs/autotune_*.log`. |
| `install_autotune_task.ps1` | Registers the **weekly** task `revbot-autotune` (Sundays), so Monday's bot starts on fresh candidates. |

## Install

From your repo root, in a normal (non-admin) PowerShell:

```powershell
# 1) Make sure the bot runs by hand first (paper account!).
python main.py        # Ctrl+C after you see it poll + respect the allowlist

# 2) Register the scheduled task.
.\scheduler\install_task.ps1            # assumes C:\revbot
# or, if your checkout is elsewhere:
.\scheduler\install_task.ps1 -RepoPath D:\path\to\revbot

# 3) Smoke-test it now instead of waiting for 08:25.
Start-ScheduledTask -TaskName revbot
Get-Content .\logs\bot_$(Get-Date -Format yyyyMMdd).log -Wait
```

## Important notes

- **Paper first.** This places real orders if `.env` points at the live
  Alpaca URL. Confirm a few *scheduled* runs on paper (it launches, trades the
  allowlist, liquidates at EOD) before switching `APCA_API_BASE_URL` to live.
- **Python path.** The task calls `run_bot.ps1`, which defaults to `python`.
  If the task can't find Python, edit the `Action` in `revbot-task.xml` (or the
  `-Python` default in `run_bot.ps1`) to the full path, e.g.
  `C:\Users\<you>\AppData\Local\Programs\Python\Python311\python.exe`.
- **Machine must be on, and — by default — logged on.** `install_task.ps1`
  imports `revbot-task.xml`, whose principal is `InteractiveToken`: the 08:25
  trigger only fires while you're actually signed in. If the box was at the
  lock screen, signed out, or asleep without a session, the bot never starts
  ("it didn't fire this morning"). To run unattended, register with
  **`install_task_headless.ps1`** instead — it stores your Windows credential
  and runs the task "whether the user is logged on or not" (same lifecycle,
  same `run_bot.ps1` wrapper), exactly like the weekly autotune task. The
  machine still has to be powered on (Task Scheduler can wake it from sleep via
  `WakeToRun`, but not from a full shutdown). For a truly always-on deployment,
  prefer an always-on server or a service manager like NSSM.
- **`.env` is read from the working directory** (`C:\revbot`), which the task
  and `run_bot.ps1` set explicitly — so your allowlist/params are picked up.
- **Time zone.** The 08:25 trigger uses the machine's local time. Adjust the
  `StartBoundary` time in `revbot-task.xml` if you're not on US Central.

## Weekly auto-tune (fresh candidates without daily noise)

The bot reads `TRADE_ALLOWLIST` + `symbol_params.json` **only at startup** — it
never re-selects intraday. To refresh candidates on a cadence, register the
weekly tuner. It runs Sundays, so Monday's 08:25 launch starts on a freshly
tuned allowlist and per-symbol params.

```powershell
# Elevated PowerShell (headless task needs it). Prompts for your Windows password.
.\scheduler\install_autotune_task.ps1 -RepoPath C:\3rev

# Smoke-test now (takes a while — full walk-forward over the universe):
Start-ScheduledTask -TaskName revbot-autotune
Get-Content .\logs\autotune_$(Get-Date -Format yyyyMMdd).log -Wait
```

- **Weekly, not daily — on purpose.** A 90-day walk-forward is stable; daily
  re-tuning would churn the universe on run-to-run noise (we've seen a single
  name's PF swing wildly between adjacent runs). Weekly adapts without chasing
  snapshots.
- **No conflict with a running bot.** The tune only writes files; the bot picks
  them up at its next (Monday) start. Running them at the same time is harmless.
- **`TRADE_BLOCKLIST` still wins.** Anything you've blocklisted stays out even
  if a tune re-allowlists it.

## Uninstall

```powershell
Unregister-ScheduledTask -TaskName revbot -Confirm:$false
Unregister-ScheduledTask -TaskName revbot-autotune -Confirm:$false
```
