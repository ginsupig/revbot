"""Session-schedule helpers (pure, network-free, unit-testable).

Kept out of main.py so the decision logic can be tested without importing the
broker SDK or the whole trading stack.
"""
from __future__ import annotations

from datetime import timedelta


def within_minutes_before_close(now_ct, minutes, close_hour: int = 15, close_minute: int = 0) -> bool:
    """True when ``now_ct`` is within the last ``minutes`` before the close.

    ``now_ct`` is a tz-aware datetime in exchange-local time (America/Chicago,
    regular 15:00 close). Used to block NEW entries late in the session so
    nothing is opened that can't be flattened before the bell — the carryover
    guard for positions that orphaned overnight when their EOD flatten missed
    the close. Pure (caller supplies ``now``); ``minutes <= 0`` disables it.
    """
    if minutes <= 0:
        return False
    close = now_ct.replace(hour=close_hour, minute=close_minute, second=0, microsecond=0)
    return (close - timedelta(minutes=minutes)) <= now_ct < close


def session_done(is_open: bool, next_open, now) -> bool:
    """Has the regular session ended for the day (with none left today)?

    ``True`` when the market is closed AND its next open falls on a *later*
    calendar day than ``now`` — i.e. we're past today's close (or it's a
    non-trading day), not merely waiting for today's open. Trusting the broker
    clock's ``next_open`` makes this correct across half-days and holidays.

    ``next_open`` and ``now`` are timezone-aware datetimes in the *same* zone
    (the caller converts both to exchange-local time first).
    """
    if is_open:
        return False
    return next_open.date() > now.date()
