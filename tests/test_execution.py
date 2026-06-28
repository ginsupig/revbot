import sys
import types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

# The executor imports the legacy alpaca_trade_api SDK at module load. We never
# construct a real REST client in these tests (we bypass __init__), so stub the
# module so the import succeeds without the dependency or any network access.
if "alpaca_trade_api" not in sys.modules:
    _stub = types.ModuleType("alpaca_trade_api")
    _rest = types.ModuleType("alpaca_trade_api.rest")
    _rest.REST = object
    _stub.rest = _rest
    sys.modules["alpaca_trade_api"] = _stub
    sys.modules["alpaca_trade_api.rest"] = _rest

from reversion_bot.execution import AlpacaExecutor
from reversion_bot.models import PositionPlan


class FakeClock:
    is_open = True


class FakePosition:
    def __init__(self, symbol, qty):
        self.symbol = symbol
        self.qty = qty


class FakeOrder:
    def __init__(self, symbol, order_id=None):
        self.symbol = symbol
        self.id = order_id if order_id is not None else f"oid-{symbol}"


class FakeClient:
    def __init__(self, positions=None, open_orders=None, close_fail_times=0):
        self._positions = positions or []
        self._open_orders = [FakeOrder(s) for s in (open_orders or [])]
        self.orders = []
        self.cancelled = []        # order ids passed to cancel_order
        self.closed = []           # symbols passed to close_position
        # How many initial close_position calls should raise before succeeding
        # (simulates held-for-orders qty freeing a beat after the cancel).
        self._close_fail_times = close_fail_times

    def get_clock(self):
        return FakeClock()

    def list_positions(self):
        return self._positions

    def list_orders(self, status=None, limit=None):
        return self._open_orders

    def submit_order(self, **kwargs):
        self.orders.append(kwargs)
        return {"id": "fake", **kwargs}

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)

    def close_position(self, symbol):
        if len(self.closed) < self._close_fail_times:
            self.closed.append(symbol)
            raise RuntimeError("insufficient qty available for order")
        self.closed.append(symbol)
        return {"id": "closed", "symbol": symbol}


def make_executor(positions=None, use_limit_entry=False, limit_offset_bps=8.0,
                  open_orders=None, close_fail_times=0):
    # Bypass __init__ so we never construct a real Alpaca REST client / hit network.
    ex = AlpacaExecutor.__new__(AlpacaExecutor)
    ex.client = FakeClient(positions, open_orders=open_orders, close_fail_times=close_fail_times)
    ex._tif = "day"
    ex._use_limit_entry = use_limit_entry
    ex._limit_offset_bps = limit_offset_bps
    return ex


def short_candidate():
    return {
        "symbol": "SPY",
        "go_short": True,
        "side": "short",
        "position_plan": {
            "qty": 10,
            "entry_price": 100.0,
            "stop_price": 101.2,    # above entry
            "target_price": 98.0,   # below entry
            "risk_per_share": 1.2,
            "reward_per_share": 2.0,
            "rr_ratio": 1.67,
            "position_value": 1000.0,
            "max_account_risk": 500.0,
            "side": "short",
        },
    }


def long_candidate():
    return {
        "symbol": "SPY",
        "go_long": True,
        "side": "long",
        "position_plan": {
            "qty": 10,
            "entry_price": 100.0,
            "stop_price": 98.0,     # below entry
            "target_price": 104.0,  # above entry
            "risk_per_share": 2.0,
            "reward_per_share": 4.0,
            "rr_ratio": 2.0,
            "position_value": 1000.0,
            "max_account_risk": 500.0,
            "side": "long",
        },
    }


def test_market_entry_by_default():
    ex = make_executor()  # use_limit_entry defaults False
    ex.submit_order(long_candidate())
    order = ex.client.orders[-1]
    assert order["type"] == "market"
    assert "limit_price" not in order


def test_limit_entry_buy_is_priced_above_entry():
    ex = make_executor(use_limit_entry=True, limit_offset_bps=8.0)
    ex.submit_order(long_candidate())
    order = ex.client.orders[-1]
    assert order["type"] == "limit"
    # Marketable buy: 8 bps above the 100.00 entry -> 100.08, so it crosses up.
    assert order["limit_price"] == 100.08
    assert order["limit_price"] > 100.0


def test_limit_entry_short_is_priced_below_entry():
    ex = make_executor(use_limit_entry=True, limit_offset_bps=8.0)
    ex.submit_order(short_candidate())
    order = ex.client.orders[-1]
    assert order["type"] == "limit"
    # Marketable short sell: 8 bps below the 100.00 entry -> 99.92, crosses down.
    assert order["limit_price"] == 99.92
    assert order["limit_price"] < 100.0


def test_submit_order_routes_short_to_sell_bracket():
    ex = make_executor()
    ex.submit_order(short_candidate())
    assert len(ex.client.orders) == 1
    order = ex.client.orders[0]
    assert order["side"] == "sell"
    assert order["order_class"] == "bracket"
    # On a short, take-profit sits below entry and the stop sits above.
    assert order["take_profit"]["limit_price"] == 98.0
    assert order["stop_loss"]["stop_price"] == 101.2


def test_short_bracket_rejects_long_geometry():
    ex = make_executor()
    bad = PositionPlan(
        qty=10, entry_price=100.0, stop_price=98.0, target_price=102.0,
        risk_per_share=2.0, reward_per_share=2.0, rr_ratio=1.0,
        position_value=1000.0, max_account_risk=500.0, side="short",
    )
    with pytest.raises(ValueError):
        ex.open_short_bracket("SPY", bad)


def test_short_blocked_when_short_already_open():
    ex = make_executor(positions=[FakePosition("SPY", "-10")])
    with pytest.raises(RuntimeError):
        ex.submit_order(short_candidate())


def test_has_open_position_detects_both_sides():
    long_ex = make_executor(positions=[FakePosition("SPY", "5")])
    short_ex = make_executor(positions=[FakePosition("SPY", "-5")])
    flat_ex = make_executor(positions=[FakePosition("SPY", "0")])
    assert long_ex.has_open_position("SPY") and long_ex.has_open_long_position("SPY")
    assert short_ex.has_open_position("SPY") and short_ex.has_open_short_position("SPY")
    assert not flat_ex.has_open_position("SPY")


def test_tune_connection_pool_mounts_larger_adapter():
    import requests
    ex = make_executor()
    ex.client._session = requests.Session()
    ex._tune_connection_pool(32)
    adapter = ex.client._session.get_adapter("https://paper-api.alpaca.markets")
    assert getattr(adapter, "_pool_maxsize", None) == 32


def test_tune_connection_pool_noop_without_session():
    ex = make_executor()  # FakeClient has no _session attribute
    ex._tune_connection_pool(32)  # must not raise


# --- Open-order reconciliation (double-submit guard) -------------------------

def test_open_order_symbols_and_has_open_order():
    ex = make_executor(open_orders=["AAPL", "spy"])
    assert ex.open_order_symbols() == {"AAPL", "SPY"}
    assert ex.has_open_order("aapl") is True
    assert ex.has_open_order("MSFT") is False


def test_long_bracket_blocked_by_working_order():
    # A working (unfilled) order on the symbol must block a new bracket even
    # though there's no filled position yet.
    ex = make_executor(open_orders=["SPY"])
    with pytest.raises(RuntimeError, match="Working order already exists"):
        ex.submit_order(long_candidate())
    assert ex.client.orders == []   # nothing submitted


def test_short_bracket_blocked_by_working_order():
    ex = make_executor(open_orders=["SPY"])
    with pytest.raises(RuntimeError, match="Working order already exists"):
        ex.submit_order(short_candidate())
    assert ex.client.orders == []


def test_no_working_order_allows_submit():
    ex = make_executor(open_orders=[])
    ex.submit_order(long_candidate())
    assert len(ex.client.orders) == 1


# --- Breakout-exit close: cancel the bracket legs before liquidating ---------

def test_close_long_cancels_symbol_orders_before_closing():
    # The held bracket legs reserve the shares; without cancelling them first
    # close_position is rejected with "insufficient qty available". Only the
    # target symbol's orders are cancelled — other names are left alone.
    ex = make_executor(open_orders=["OXY", "OXY", "AAPL"])
    ex.close_long("oxy")
    assert ex.client.cancelled == ["oid-OXY", "oid-OXY"]   # AAPL untouched
    assert ex.client.closed == ["OXY"]


def test_close_long_retries_close_until_qty_frees(monkeypatch):
    # The cancelled qty can take a beat to free, so the first close may still
    # raise; close_long retries rather than erroring out for the whole cycle.
    monkeypatch.setattr("reversion_bot.execution.sleep", lambda *_: None)
    ex = make_executor(open_orders=["OXY"], close_fail_times=2)
    ex.close_long("OXY")
    assert ex.client.cancelled == ["oid-OXY"]
    assert ex.client.closed == ["OXY", "OXY", "OXY"]   # 2 failures + 1 success


def test_close_long_survives_order_list_failure():
    # A list_orders hiccup must not block the close — it still attempts to close.
    ex = make_executor(open_orders=["OXY"])
    ex.client.list_orders = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    ex.close_long("OXY")
    assert ex.client.cancelled == []
    assert ex.client.closed == ["OXY"]
