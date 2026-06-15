import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reversion_bot.trailing import trailing_stop_price, should_raise_stop


# entry 100, stop_mult 1.2, trail_mult 1.5, stop_distance = 1.2*ATR.
# With ATR=1 -> stop_distance=1.2, init_stop=98.8, trail_dist=1.5.

def test_floored_at_initial_stop_before_price_rises():
    # high_water == entry: candidate = 100-1.5 = 98.5 < init_stop 98.8 -> floored.
    s = trailing_stop_price(100.0, 1.2, high_water=100.0, stop_atr_mult=1.2, trail_atr_mult=1.5)
    assert abs(s - 98.8) < 1e-9


def test_ratchets_up_once_in_profit():
    # high_water 103: candidate = 103-1.5 = 101.5 > init_stop -> trail governs.
    s = trailing_stop_price(100.0, 1.2, high_water=103.0, stop_atr_mult=1.2, trail_atr_mult=1.5)
    assert abs(s - 101.5) < 1e-9


def test_locks_profit_above_entry():
    # A big run: stop trails into profit (above entry).
    s = trailing_stop_price(100.0, 1.2, high_water=105.0, stop_atr_mult=1.2, trail_atr_mult=1.5)
    assert s == 103.5 and s > 100.0


def test_trail_distance_scales_with_atr():
    # ATR=2 -> stop_distance=2.4, trail_dist = 1.5/1.2 * 2.4 = 3.0.
    s = trailing_stop_price(100.0, 2.4, high_water=110.0, stop_atr_mult=1.2, trail_atr_mult=1.5)
    assert abs(s - 107.0) < 1e-9   # 110 - 3.0


def test_should_raise_only_upward_and_below_price():
    assert should_raise_stop(98.8, 101.5, last_price=103.0) is True
    assert should_raise_stop(101.5, 101.5, last_price=103.0) is False   # no change
    assert should_raise_stop(101.5, 101.0, last_price=103.0) is False   # would move down
    assert should_raise_stop(98.8, 99.0, last_price=103.0, min_increment=0.5) is False  # too small


def test_should_raise_never_at_or_above_last_price():
    # Desired stop above market would trigger instantly -> blocked.
    assert should_raise_stop(98.8, 103.5, last_price=103.0) is False
    assert should_raise_stop(98.8, 103.0, last_price=103.0) is False
