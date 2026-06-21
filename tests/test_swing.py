import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime
from zoneinfo import ZoneInfo

from reversion_bot.swing import (
    is_moc_entry_window, should_block_entry_now, holds_overnight,
)

CT = ZoneInfo("America/Chicago")


def _ct(h, m):
    return datetime(2025, 6, 2, h, m, tzinfo=CT)      # a regular weekday


def test_moc_window_default_2_30_to_2_50_ct():
    # default window_min=20 -> [2:30, 2:50) CT (ends at the 3:50 ET MOC deadline)
    assert is_moc_entry_window(_ct(14, 30), 20) is True
    assert is_moc_entry_window(_ct(14, 45), 20) is True
    assert is_moc_entry_window(_ct(14, 29), 20) is False   # just before the window
    assert is_moc_entry_window(_ct(14, 50), 20) is False   # at the deadline -> too late
    assert is_moc_entry_window(_ct(10, 0), 20) is False    # mid-morning


def test_moc_window_respects_window_length():
    assert is_moc_entry_window(_ct(14, 41), 10) is True    # [2:40, 2:50)
    assert is_moc_entry_window(_ct(14, 39), 10) is False


def test_should_block_entry_only_in_swing_outside_window():
    # swing mode blocks entries OUTSIDE the MOC window...
    assert should_block_entry_now(swing_mode=True, in_moc_window=False) is True
    # ...and allows them inside it
    assert should_block_entry_now(swing_mode=True, in_moc_window=True) is False
    # intraday mode never blocks here (its own EOD cutoff handles late entries)
    assert should_block_entry_now(swing_mode=False, in_moc_window=False) is False
    assert should_block_entry_now(swing_mode=False, in_moc_window=True) is False


def test_holds_overnight_tracks_flag():
    assert holds_overnight(True) is True
    assert holds_overnight(False) is False
