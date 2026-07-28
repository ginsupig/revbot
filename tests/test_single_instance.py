import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reversion_bot.single_instance import acquire_lock, release_lock, _pid_alive


def test_acquire_then_release(tmp_path):
    lock = tmp_path / "revbot.lock"
    ok, pid = acquire_lock(lock)
    assert ok is True
    assert pid == os.getpid()
    assert lock.read_text().splitlines()[0] == str(os.getpid())

    release_lock(lock)
    assert not lock.exists()


def test_second_acquire_blocked_by_live_holder(tmp_path):
    lock = tmp_path / "revbot.lock"
    # Simulate a live holder = our own PID (definitely alive).
    lock.write_text(str(os.getpid()))

    # A different "instance" can't claim it while our PID holds it. We emulate a
    # foreign caller by checking the live-holder branch directly: since the file
    # holds the current pid, acquire treats it as ours and reclaims — so to test
    # the *blocked* path we write a PID that is alive but not us is impossible
    # here; instead assert the live-pid detection itself.
    assert _pid_alive(os.getpid()) is True


def test_stale_lock_is_reclaimed(tmp_path):
    lock = tmp_path / "revbot.lock"
    # A PID that is essentially guaranteed not to exist.
    lock.write_text("999999")
    assert _pid_alive(999999) is False

    ok, pid = acquire_lock(lock)
    assert ok is True
    assert pid == os.getpid()
    assert lock.read_text().splitlines()[0] == str(os.getpid())
    release_lock(lock)


def test_release_only_if_owner(tmp_path):
    lock = tmp_path / "revbot.lock"
    lock.write_text("12345")  # someone else owns it
    release_lock(lock)        # must NOT delete a lock we don't own
    assert lock.exists()
    assert lock.read_text().strip() == "12345"


def test_dead_pid_detection():
    assert _pid_alive(0) is False
    assert _pid_alive(-1) is False
    assert _pid_alive(os.getpid()) is True


def test_live_foreign_holder_is_refused(tmp_path, monkeypatch):
    # A different, *alive* process holds the lock -> acquire must refuse and
    # leave the existing lock untouched (the core of the single-instance safety).
    import reversion_bot.single_instance as si

    lock = tmp_path / "revbot.lock"
    lock.write_text("424242")
    monkeypatch.setattr(si, "_pid_alive", lambda pid: pid == 424242)

    ok, holder = si.acquire_lock(lock)
    assert ok is False
    assert holder == 424242
    assert lock.read_text().strip() == "424242"   # foreign lock preserved


def test_atomic_acquire_is_exclusive(tmp_path):
    # First caller wins; acquiring again returns held-by-us rather than silently
    # granting a second independent lock or clearing the owner's file.
    lock = tmp_path / "revbot.lock"
    ok1, pid1 = acquire_lock(lock)
    assert ok1 is True and pid1 == os.getpid()

    ok2, pid2 = acquire_lock(lock)
    assert ok2 is True and pid2 == os.getpid()
    assert lock.read_text().splitlines()[0] == str(os.getpid())

    release_lock(lock)
    assert not lock.exists()


# --- PID reuse after a reboot (audit C5) -------------------------------------
# The lock records the holder's process START TIME; a live pid whose start
# time doesn't match was recycled by the OS and must be treated as stale —
# the old behavior refused forever (with exit code 0: a silent trading halt).

def _spawn_sleeper():
    import subprocess, sys as _sys
    return subprocess.Popen([_sys.executable, "-c", "import time; time.sleep(30)"])


def test_recycled_pid_is_treated_as_stale(tmp_path):
    from reversion_bot.single_instance import _pid_start_time

    proc = _spawn_sleeper()
    try:
        assert _pid_alive(proc.pid) is True
        # Lock claims this live pid started at unix time 12345 — i.e. the
        # recorded holder was a DIFFERENT process that happened to have the
        # same pid. Must reclaim.
        lock = tmp_path / "revbot.lock"
        lock.write_text(f"{proc.pid}\n12345.0")
        ok, pid = acquire_lock(lock)
        assert ok is True
        assert pid == os.getpid()
        release_lock(lock)
    finally:
        proc.kill()
        proc.wait()


def test_genuine_live_holder_with_matching_start_time_is_refused(tmp_path):
    from reversion_bot.single_instance import _pid_start_time

    proc = _spawn_sleeper()
    try:
        start = _pid_start_time(proc.pid)
        if start is None:
            import pytest
            pytest.skip("process start time unavailable on this platform")
        lock = tmp_path / "revbot.lock"
        lock.write_text(f"{proc.pid}\n{start}")
        ok, holder = acquire_lock(lock)
        assert ok is False
        assert holder == proc.pid
        assert lock.read_text().splitlines()[0] == str(proc.pid)
    finally:
        proc.kill()
        proc.wait()


def test_own_start_time_is_measurable_and_recorded(tmp_path):
    from reversion_bot.single_instance import _pid_start_time
    import time

    start = _pid_start_time(os.getpid())
    if start is None:
        import pytest
        pytest.skip("process start time unavailable on this platform")
    assert 0 < start <= time.time() + 1

    lock = tmp_path / "revbot.lock"
    ok, _ = acquire_lock(lock)
    assert ok is True
    lines = lock.read_text().splitlines()
    assert lines[0] == str(os.getpid())
    assert len(lines) == 2 and abs(float(lines[1]) - start) < 3.0
    release_lock(lock)
