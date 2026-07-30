import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

# The execution module imports the alpaca SDK at import time; skip cleanly where
# it isn't installed (runs in CI, which has it).
pytest.importorskip("alpaca_trade_api")
from reversion_bot.execution import AlpacaExecutor


class _Order:
    def __init__(self, symbol, oid, stop_price=None, qty=10):
        self.symbol, self.id, self.stop_price, self.qty = symbol, oid, stop_price, qty


class _FakeClient:
    def __init__(self, orders):
        self._orders = orders
        self.replaced = []

    def list_orders(self, status="open", limit=500):
        return self._orders

    def replace_order(self, order_id, stop_price=None, limit_price=None):
        self.replaced.append((order_id, stop_price, limit_price))
        return {"id": order_id, "stop_price": stop_price}


def _exec(orders):
    ex = AlpacaExecutor.__new__(AlpacaExecutor)   # bypass __init__/network
    ex.client = _FakeClient(orders)
    return ex


def test_open_stop_leg_finds_the_stop_not_the_limit():
    ex = _exec([_Order("WDC", "lim", stop_price=None), _Order("WDC", "stp", stop_price=98.8)])
    leg = ex.open_stop_leg("WDC")
    assert leg["id"] == "stp" and leg["stop_price"] == 98.8


def test_update_trailing_raises_stop_when_in_profit():
    ex = _exec([_Order("WDC", "stp", stop_price=98.8)])
    # entry 100, stop_dist 1.2 (ATR 1), high_water 103 -> desired 101.5, below price 103.
    new = ex.update_trailing_stop("WDC", 100.0, 1.2, 103.0, 103.0, 1.2, 1.5)
    assert new == 101.5
    assert ex.client.replaced == [("stp", 101.5, None)]


def test_update_trailing_noop_when_not_in_profit():
    ex = _exec([_Order("WDC", "stp", stop_price=98.8)])
    # high_water == entry -> desired floored at 98.8 == current -> no move.
    assert ex.update_trailing_stop("WDC", 100.0, 1.2, 100.0, 100.0, 1.2, 1.5) is None
    assert ex.client.replaced == []


def test_update_trailing_noop_when_no_stop_leg():
    ex = _exec([_Order("WDC", "lim", stop_price=None)])
    assert ex.update_trailing_stop("WDC", 100.0, 1.2, 103.0, 103.0, 1.2, 1.5) is None


# --- short-side wiring (audit: the short executor path had no broker test) ---

def test_update_trailing_stop_short_lowers_the_stop_when_in_profit():
    """Mirror of the long path: a short in profit ratchets its stop DOWN.

    entry 100, stop_dist 1.2 (ATR 1.0 at 1.2x), low_water 97 -> desired stop
    97 + 1.5*1.0 = 98.5, which is below the existing 101.2 and above the market.
    """
    ex = _exec([_Order("WDC", "stp", stop_price=101.2)])
    new_stop = ex.update_trailing_stop_short(
        "WDC", entry_price=100.0, stop_distance=1.2, low_water=97.0,
        last_price=97.0, stop_mult=1.2, trail_mult=1.5,
    )
    assert new_stop == pytest.approx(98.5)
    assert ex.client.replaced == [("stp", 98.5, None)]


def test_update_trailing_stop_short_never_raises_the_stop():
    """A short's stop must only move DOWN — a higher 'trail' would widen risk."""
    ex = _exec([_Order("WDC", "stp", stop_price=98.0)])
    new_stop = ex.update_trailing_stop_short(
        "WDC", entry_price=100.0, stop_distance=1.2, low_water=99.5,
        last_price=99.5, stop_mult=1.2, trail_mult=1.5,
    )
    assert new_stop is None
    assert ex.client.replaced == []


def test_update_trailing_stop_short_no_stop_leg_is_a_noop():
    ex = _exec([_Order("WDC", "lim", stop_price=None)])
    assert ex.update_trailing_stop_short(
        "WDC", entry_price=100.0, stop_distance=1.2, low_water=97.0,
        last_price=97.0, stop_mult=1.2, trail_mult=1.5,
    ) is None
    assert ex.client.replaced == []
