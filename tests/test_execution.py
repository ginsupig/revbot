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


class FakeClient:
    def __init__(self, positions=None):
        self._positions = positions or []
        self.orders = []

    def get_clock(self):
        return FakeClock()

    def list_positions(self):
        return self._positions

    def submit_order(self, **kwargs):
        self.orders.append(kwargs)
        return {"id": "fake", **kwargs}


def make_executor(positions=None):
    # Bypass __init__ so we never construct a real Alpaca REST client / hit network.
    ex = AlpacaExecutor.__new__(AlpacaExecutor)
    ex.client = FakeClient(positions)
    ex._order_ids = set()
    ex._tif = "day"
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
