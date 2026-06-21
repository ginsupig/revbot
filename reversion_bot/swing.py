"""Swing / overnight-hold mode for the validated daily broad-reversion strategy.

The research arc (research/portfolio_sim.py) landed a deployable, two-window-robust
edge: a daily oversold-reversion basket on the neutral large-cap universe, entered
at the CLOSE (market-on-close), held overnight and multi-day, with the ATR bracket
+ trailing stop cutting losers and riding winners. The 2025-26 window's edge lives
almost entirely in the close->next-open gap, so entering at the close — not intraday
and not next-open — is load-bearing (+20.6% vs -2.0% on that window).

The live bot is intraday and flat-by-the-bell, so this mode INVERTS three behaviors,
all behind a default-off flag (SWING_OVERNIGHT_MODE):
  * enter only in the market-on-close submission window (not all day),
  * do NOT flatten at the EOD window / session end — hold overnight,
  * do NOT carryover-flatten the bot's own positions next morning.
The exit stack (ATR bracket + trailing stop) is unchanged, and the order itself is
made market-on-close via the existing TIF knob (TIF=cls). Default OFF reproduces
today's intraday, flat-overnight behavior exactly.

Pure decision logic so it is unit-testable; main.py reads the env flag and wires
these into the loop.
"""
from __future__ import annotations

from datetime import datetime, timedelta


def is_moc_entry_window(now_ct: datetime, window_min: int,
                        cutoff_hour: int = 14, cutoff_minute: int = 50) -> bool:
    """True inside the market-on-close submission window ``[cutoff - window_min, cutoff)``.

    ``cutoff`` defaults to 2:50 PM CT (3:50 PM ET) — the broker MOC deadline — so a
    close entry is submitted in time to fill at the closing print rather than after
    the deadline. Naive comparison on ``now_ct`` (already America/Chicago)."""
    cutoff = now_ct.replace(hour=cutoff_hour, minute=cutoff_minute, second=0, microsecond=0)
    start = cutoff - timedelta(minutes=max(window_min, 0))
    return start <= now_ct < cutoff


def should_block_entry_now(swing_mode: bool, in_moc_window: bool) -> bool:
    """Swing mode trades once/day at the close: block new entries OUTSIDE the MOC
    window. Normal mode returns ``False`` here (its own EOD cutoff handles late
    entries) — so this never changes non-swing behavior."""
    return bool(swing_mode and not in_moc_window)


def holds_overnight(swing_mode: bool) -> bool:
    """Whether the bot holds positions overnight/multi-day (exit stack manages them).
    When True, EOD-flatten, session-end flatten and carryover-flatten are all skipped."""
    return bool(swing_mode)
