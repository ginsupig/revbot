"""Single-instance guard for the live bot.

Stops a second `python main.py` from running against the *same checkout* (and
therefore the same account/config), which would double-place orders and blow
through the risk limits. The lock file lives in the project's state dir, so two
*different* checkouts (e.g. a separate paper deployment) each get their own
lock and can run side by side — only a duplicate of the same deployment is
refused.

A lock left behind by a hard kill (Stop-Process / SIGKILL skip cleanup) is
detected as stale via a process-liveness check and reclaimed on next start.
"""
from __future__ import annotations

import atexit
import os
from pathlib import Path
from typing import Tuple


def _pid_alive(pid: int) -> bool:
    """Best-effort cross-platform check that a PID is a running process."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return code.value == STILL_ACTIVE
            return True  # couldn't read exit code; assume alive
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just not ours to signal
    return True


def _read_pid(path: Path) -> int:
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, ValueError, OSError):
        return 0


def acquire_lock(lock_path: "str | os.PathLike") -> Tuple[bool, int]:
    """Try to claim the single-instance lock, atomically.

    Returns ``(acquired, holder_pid)``. Uses an *exclusive create*
    (``O_CREAT | O_EXCL``) so that when two launches race, exactly one wins and
    the loser is refused — closing the read-then-write gap that let manual and
    scheduled starts stack up. A lock left by a dead process is detected as
    stale and reclaimed. On refusal, ``holder_pid`` is the live process that
    already holds it.
    """
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mypid = os.getpid()

    # At most one stale-reclaim, then one retry of the exclusive create.
    for _ in range(2):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder = _read_pid(path)
            if holder == mypid:
                # Already ours (e.g. re-entrant call) — treat as held.
                atexit.register(release_lock, str(path))
                return True, mypid
            if holder and _pid_alive(holder):
                return False, holder          # someone else is live — refuse
            # Stale (dead/unreadable holder): drop it and retry the create.
            try:
                os.unlink(str(path))
            except FileNotFoundError:
                pass
            continue
        else:
            with os.fdopen(fd, "w") as f:
                f.write(str(mypid))
            atexit.register(release_lock, str(path))
            return True, mypid

    # Couldn't claim even after a reclaim attempt (another racer won) — refuse.
    holder = _read_pid(path)
    return False, holder or mypid


def release_lock(lock_path: "str | os.PathLike") -> None:
    """Release the lock, but only if this process actually owns it."""
    path = Path(lock_path)
    if _read_pid(path) == os.getpid():
        try:
            path.unlink()
        except FileNotFoundError:
            pass
