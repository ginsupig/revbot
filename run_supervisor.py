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


def _log(msg: str) -> None:
    print(f"[SUPERVISOR {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    fast_exit_backoff = 30      # seconds to wait after a crash (fast exit)
    normal_relaunch = 20        # seconds to wait after a clean exit (e.g. at the close)
    fast_exit_threshold = 60    # a run shorter than this is treated as a crash

    _log("starting. Launching main.py; will relaunch on exit. Ctrl+C to stop.")
    while True:
        started = time.monotonic()
        try:
            rc = subprocess.run([sys.executable, "main.py"]).returncode
        except KeyboardInterrupt:
            _log("Ctrl+C — stopping supervisor.")
            return 0
        except Exception as e:                     # never let the supervisor itself die
            _log(f"launch error: {e!r} — retrying in {fast_exit_backoff}s")
            time.sleep(fast_exit_backoff)
            continue

        ran_for = time.monotonic() - started
        if ran_for < fast_exit_threshold:
            _log(f"bot exited fast ({ran_for:.0f}s, rc={rc}) — likely an error. "
                 f"Backing off {fast_exit_backoff}s.")
            time.sleep(fast_exit_backoff)
        else:
            _log(f"bot exited cleanly (ran {ran_for/60:.0f}m, rc={rc}) — relaunching "
                 f"in {normal_relaunch}s so it's alive for the next session.")
            time.sleep(normal_relaunch)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[SUPERVISOR] stopped.")
