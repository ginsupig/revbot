"""Session-schedule helpers (pure, network-free, unit-testable).

Kept out of main.py so the decision logic can be tested without importing the
broker SDK or the whole trading stack.
"""
from __future__ import annotations


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
