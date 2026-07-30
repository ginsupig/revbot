"""Supervisor: keep the bot alive so a missed scheduler fire never costs a session.

The swing/MOC bot enters in a 20-minute window and exits at the close, so it MUST
be running at 2:30-2:50pm CT. Relying on the OS scheduler to fire correctly every
day is fragile (machine asleep, "run only when logged on", wrong start-dir, a stale
lock) — a single missed fire = a whole day of no trades.

This removes that dependency: launch it ONCE and leave it running. It relaunches
main.py whenever the bot exits (the bot sleeps by itself pre-open, so this is not a
busy loop), with backoff if the bot dies fast (a real error) so it can't hot-loop.
The only remaining requirement is that this process stays alive — so run it on a
box that does NOT sleep (disable sleep, or use an always-on / cloud host).

    python run_supervisor.py

Stop with Ctrl+C.
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Resolve main.py next to this file, not relative to the launch cwd.
_MAIN_SCRIPT = Path(__file__).resolve().parent / "main.py"


def _log(msg: str) -> None:
    print(f"[SUPERVISOR {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    crash_backoff = 60          # after a CRASH (rc != 0): relaunch soon
    clean_backoff = 900         # after a CLEAN exit (rc == 0, e.g. market-closed): wait
    long_run_relaunch = 20      # after a long run that then ended: relaunch promptly

    _log("starting. Launching main.py; will relaunch on exit. Ctrl+C to stop.")
    _log("NOTE: prefer RUN_PERSISTENT=true — then main.py never exits and this "
         "supervisor is just crash-insurance (no relaunch spam).")
    while True:
        started = time.monotonic()
        try:
            # Absolute path: launching "main.py" relative to the cwd is exactly
            # the wrong-start-dir failure this supervisor exists to survive —
            # it would exit rc=2 ("can't open file"), be classified a crash, and
            # relaunch nothing every 60s forever.
            rc = subprocess.run([sys.executable, str(_MAIN_SCRIPT)]).returncode
        except KeyboardInterrupt:
            _log("Ctrl+C — stopping supervisor.")
            return 0
        except Exception as e:                     # never let the supervisor itself die
            _log(f"launch error: {e!r} — retrying in {crash_backoff}s")
            time.sleep(crash_backoff)
            continue

        ran_for = time.monotonic() - started
        if rc != 0:
            _log(f"bot CRASHED (ran {ran_for:.0f}s, rc={rc}) — relaunching in {crash_backoff}s.")
            time.sleep(crash_backoff)
        elif ran_for < 60:
            # Clean, fast exit = the bot chose to stop (e.g. market closed, non-
            # persistent). Do NOT hot-loop — wait a long while. (Set RUN_PERSISTENT
            # so the bot stays up instead and this branch never fires.)
            _log(f"bot exited cleanly & fast ({ran_for:.0f}s) — it chose to stop "
                 f"(market closed / non-persistent). Waiting {clean_backoff}s. "
                 f"Set RUN_PERSISTENT=true to keep it alive instead.")
            time.sleep(clean_backoff)
        else:
            _log(f"bot exited after a long run ({ran_for/60:.0f}m, rc=0) — "
                 f"relaunching in {long_run_relaunch}s.")
            time.sleep(long_run_relaunch)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[SUPERVISOR] stopped.")
