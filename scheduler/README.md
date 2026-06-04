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
| `install_task.ps1` | Registers/updates the task named `revbot`. |

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
- **Machine must be on.** A per-user task only runs while your PC is on (and,
  as written, while you're logged on — `InteractiveToken`). For a true
  always-on box that survives logoff/reboot unattended, run it on an
  always-on server or switch to a service manager like NSSM.
- **`.env` is read from the working directory** (`C:\revbot`), which the task
  and `run_bot.ps1` set explicitly — so your allowlist/params are picked up.
- **Time zone.** The 08:25 trigger uses the machine's local time. Adjust the
  `StartBoundary` time in `revbot-task.xml` if you're not on US Central.

## Uninstall

```powershell
Unregister-ScheduledTask -TaskName revbot -Confirm:$false
```
