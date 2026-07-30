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
import logging
import os
import time
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


def _pid_start_time(pid: int) -> "float | None":
    """Process creation time in unix seconds, or None when unavailable.

    This is what tells a REUSED pid apart from the original lock holder: after
    a reboot (or plain pid recycling) the OS hands the old pid to an unrelated
    process, `_pid_alive` says True, and without this check the bot refuses to
    start forever — with exit code 0, which the supervisor treats as a clean
    stop. A silent, permanent trading halt from the very mechanism meant to
    survive crashes.
    """
    try:
        if os.name == "nt":
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return None
            try:
                creation = ctypes.c_ulonglong()
                exit_t = ctypes.c_ulonglong()
                kern = ctypes.c_ulonglong()
                user = ctypes.c_ulonglong()
                ok = kernel32.GetProcessTimes(
                    handle,
                    ctypes.byref(creation),
                    ctypes.byref(exit_t),
                    ctypes.byref(kern),
                    ctypes.byref(user),
                )
                if not ok:
                    return None
                # FILETIME: 100ns ticks since 1601-01-01 UTC -> unix seconds.
                return creation.value / 1e7 - 11644473600.0
            finally:
                kernel32.CloseHandle(handle)
        stat_path = f"/proc/{pid}/stat"
        if os.path.exists(stat_path):
            with open(stat_path) as f:
                stat = f.read()
            # starttime is field 22; the comm field before it may contain
            # spaces/parens, so split on the LAST ')'.
            fields = stat.rsplit(")", 1)[1].split()
            ticks = float(fields[19])
            clk = float(os.sysconf("SC_CLK_TCK"))
            with open("/proc/stat") as f:
                for line in f:
                    if line.startswith("btime"):
                        return float(line.split()[1]) + ticks / clk
    except Exception:
        return None
    return None


# Tolerance when comparing recorded vs. actual process start time: /proc
# tick math and FILETIME rounding can differ by a hair; pid recycling across
# a reboot differs by minutes-to-days.
_START_TIME_TOLERANCE_S = 3.0

# A lock file whose pid isn't readable yet is presumed to be a racer's in-flight
# write (see acquire_lock); re-read a few times before calling it corrupt.
_INFLIGHT_RETRIES = 5
_INFLIGHT_WAIT_S = 0.05


def _read_lock(path: Path) -> Tuple[int, "float | None"]:
    """Parse the lock file: line 1 = pid, line 2 (optional) = start time.

    Legacy single-line pid files parse with start None (identity check
    skipped, pre-fix behavior)."""
    try:
        lines = path.read_text().strip().splitlines()
        pid = int(lines[0].strip())
        start = float(lines[1].strip()) if len(lines) > 1 else None
        return pid, start
    except (FileNotFoundError, ValueError, IndexError, OSError):
        return 0, None


def _read_pid(path: Path) -> int:
    return _read_lock(path)[0]


def _holder_is_live(pid: int, recorded_start: "float | None") -> bool:
    """The recorded holder is genuinely alive: pid running AND (when both
    sides are measurable) its start time matches the one recorded at lock
    time. A start-time mismatch means the pid was recycled -> stale."""
    if not _pid_alive(pid):
        return False
    if recorded_start is None:
        return True   # legacy lock: no identity to compare
    actual = _pid_start_time(pid)
    if actual is None:
        return True   # can't measure: fail safe (refuse to double-run)
    return abs(actual - recorded_start) < _START_TIME_TOLERANCE_S


def _write_lock(fd: int) -> None:
    mypid = os.getpid()
    start = _pid_start_time(mypid)
    with os.fdopen(fd, "w") as f:
        f.write(f"{mypid}\n{start if start is not None else ''}".rstrip())


def acquire_lock(lock_path: "str | os.PathLike") -> Tuple[bool, int]:
    """Try to claim the single-instance lock, atomically.

    Returns ``(acquired, holder_pid)``. Uses an *exclusive create*
    (``O_CREAT | O_EXCL``) so that when two launches race, exactly one wins and
    the loser is refused — closing the read-then-write gap that let manual and
    scheduled starts stack up. A lock left by a dead process is detected as
    stale and reclaimed; the lock records the holder's process START TIME so a
    recycled pid (same number, different process, e.g. after a reboot) is also
    treated as stale instead of halting trading forever. On refusal,
    ``holder_pid`` is the live process that already holds it.
    """
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mypid = os.getpid()

    # At most one stale-reclaim, then one retry of the exclusive create.
    for _ in range(2):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder, recorded_start = _read_lock(path)
            if not holder:
                # The file exists but carries no readable pid. The overwhelmingly
                # likely cause is a racer that just won the exclusive create and
                # hasn't written its pid yet (open and write are not atomic) —
                # treating that as "stale" would unlink the winner's fresh lock
                # and let BOTH instances run, double-placing orders. Give the
                # writer a moment to finish; only a lock that STAYS unreadable is
                # genuinely corrupt and reclaimable.
                for _ in range(_INFLIGHT_RETRIES):
                    time.sleep(_INFLIGHT_WAIT_S)
                    holder, recorded_start = _read_lock(path)
                    if holder:
                        break
                if not holder:
                    logging.warning(
                        "Lock file %s has no readable pid after %.2fs — treating "
                        "as corrupt and reclaiming.",
                        path, _INFLIGHT_RETRIES * _INFLIGHT_WAIT_S,
                    )
            if holder == mypid:
                # Already ours (e.g. re-entrant call) — treat as held.
                atexit.register(release_lock, str(path))
                return True, mypid
            if holder and _holder_is_live(holder, recorded_start):
                return False, holder          # someone else is live — refuse
            # Stale (dead/recycled/unreadable holder): drop it and retry the
            # create. Re-read immediately before the unlink so we never remove
            # a FRESH lock another racer just wrote in the gap (the reclaim
            # race: both starters judge the same holder dead, one recreates,
            # the other must not unlink the winner's new lock).
            if _read_pid(path) not in (holder, 0):
                return False, _read_pid(path) or mypid
            try:
                os.unlink(str(path))
            except FileNotFoundError:
                pass
            continue
        else:
            _write_lock(fd)
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
