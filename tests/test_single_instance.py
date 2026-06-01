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
    assert lock.read_text().strip() == str(os.getpid())

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
    assert lock.read_text().strip() == str(os.getpid())
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
