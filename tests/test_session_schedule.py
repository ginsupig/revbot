import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reversion_bot.schedule import session_done

CT = ZoneInfo("America/Chicago")


def test_open_market_is_never_done():
    now = datetime(2026, 6, 4, 10, 0, tzinfo=CT)
    nxt = datetime(2026, 6, 4, 8, 30, tzinfo=CT)
    assert session_done(True, nxt, now) is False


def test_pre_open_same_day_not_done():
    # 08:25, before the 08:30 open — next open is later today, so keep waiting.
    now = datetime(2026, 6, 4, 8, 25, tzinfo=CT)
    nxt = datetime(2026, 6, 4, 8, 30, tzinfo=CT)
    assert session_done(False, nxt, now) is False


def test_after_close_is_done():
    # 15:05, after the 15:00 close — next open is tomorrow -> session over.
    now = datetime(2026, 6, 4, 15, 5, tzinfo=CT)
    nxt = datetime(2026, 6, 5, 8, 30, tzinfo=CT)
    assert session_done(False, nxt, now) is True


def test_weekend_is_done():
    # Saturday, next open Monday -> done (no session today).
    now = datetime(2026, 6, 6, 11, 0, tzinfo=CT)
    nxt = datetime(2026, 6, 8, 8, 30, tzinfo=CT)
    assert session_done(False, nxt, now) is True


def test_half_day_after_early_close_is_done():
    # Early close at noon; it's 12:30 and the next open is tomorrow.
    now = datetime(2026, 11, 27, 12, 30, tzinfo=CT)
    nxt = datetime(2026, 11, 30, 8, 30, tzinfo=CT)
    assert session_done(False, nxt, now) is True
