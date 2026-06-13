import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reversion_bot.schedule import session_done, within_minutes_before_close

CT = ZoneInfo("America/Chicago")


# --- within_minutes_before_close (EOD entry cutoff) -------------------------

def test_inside_cutoff_window_blocks_entries():
    # 14:58, 2 min before the 15:00 close, cutoff 20 min -> True (block).
    assert within_minutes_before_close(datetime(2026, 6, 11, 14, 58, tzinfo=CT), 20) is True


def test_before_cutoff_window_allows_entries():
    # 14:30 with a 20-min cutoff (window starts 14:40) -> not yet in it.
    assert within_minutes_before_close(datetime(2026, 6, 11, 14, 30, tzinfo=CT), 20) is False


def test_at_or_after_close_is_not_in_window():
    # The window is [close-minutes, close); 15:00 and later are outside it.
    assert within_minutes_before_close(datetime(2026, 6, 11, 15, 0, tzinfo=CT), 20) is False
    assert within_minutes_before_close(datetime(2026, 6, 11, 15, 5, tzinfo=CT), 20) is False


def test_zero_minutes_disables_cutoff():
    assert within_minutes_before_close(datetime(2026, 6, 11, 14, 58, tzinfo=CT), 0) is False


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
