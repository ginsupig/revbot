"""Fetch-window math: the requested calendar window must cover the lookback.

Regression tests for the audit's C3: the old heuristic
(bars_per_day = 78 if "5" in tf else 390) gave TRADE_TIMEFRAME=1Day a
3-calendar-day window against a 160-bar lookback (~2 bars fetched), so daily/
swing mode — and the channel exit's daily fetch — silently never had enough
history to evaluate anything. 15Min matched the "5" branch (78 vs actual 26),
and 30Min/1Hour assumed 390 bars/day.
"""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# main.py imports the executor, which imports the legacy SDK at module load;
# stub it so this file also collects standalone in dependency-free sandboxes.
if "alpaca_trade_api" not in sys.modules:
    _stub = types.ModuleType("alpaca_trade_api")
    _rest = types.ModuleType("alpaca_trade_api.rest")
    _rest.REST = object
    _stub.rest = _rest
    sys.modules["alpaca_trade_api"] = _stub
    sys.modules["alpaca_trade_api.rest"] = _rest

from main import bars_per_trading_day, get_fetch_days


@pytest.mark.parametrize("tf,expected", [
    ("1Min", 390),
    ("Minute", 390),
    ("5Min", 78),
    ("15Min", 26),
    ("30Min", 13),
    ("1Hour", 6),
    ("Hour", 6),
    ("1Day", 1),
    ("Day", 1),
    ("daily", 1),
])
def test_bars_per_trading_day(tf, expected):
    assert bars_per_trading_day(tf) == expected


@pytest.mark.parametrize("tf", ["1Min", "5Min", "15Min", "30Min", "1Hour", "1Day"])
@pytest.mark.parametrize("lookback", [60, 160, 200])
def test_window_covers_lookback(tf, lookback):
    days = get_fetch_days(tf, lookback)
    # ~5 trading days per 7 calendar days; the window must span enough
    # trading days to produce `lookback` bars.
    trading_days_available = days * 5 / 7
    bars_available = trading_days_available * bars_per_trading_day(tf)
    assert bars_available >= lookback, (
        f"{tf} lookback={lookback}: {days} calendar days -> "
        f"~{bars_available:.0f} bars < {lookback}"
    )


def test_daily_swing_regression():
    # The exact broken case: 1Day/160 used to return 3 calendar days.
    assert get_fetch_days("1Day", 160) >= 229


def test_unknown_timeframe_falls_back_conservatively():
    assert bars_per_trading_day("weird") == 78
    assert get_fetch_days("weird", 160) >= 5
