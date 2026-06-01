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
    """Try to claim the single-instance lock.

    Returns ``(acquired, holder_pid)``. On success, writes our PID and registers
    an atexit hook to release it; a stale lock (holder no longer alive) is
    reclaimed. On failure, ``holder_pid`` is the live process already holding it.
    """
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = _read_pid(path)
    if existing and existing != os.getpid() and _pid_alive(existing):
        return False, existing

    path.write_text(str(os.getpid()))
    atexit.register(release_lock, str(path))
    return True, os.getpid()


def release_lock(lock_path: "str | os.PathLike") -> None:
    """Release the lock, but only if this process actually owns it."""
    path = Path(lock_path)
    if _read_pid(path) == os.getpid():
        try:
            path.unlink()
        except FileNotFoundError:
            pass
