"""The opt-in alpaca-py adapter must translate the legacy REST call surface
faithfully. These tests construct everything offline (no network, no real keys)
and verify both the order-request translation and the method routing.
"""
import sys
import types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

# The executor module imports the legacy (EOL, uninstalled) alpaca_trade_api at
# load time. The two executor-wiring tests below import it; stub the legacy SDK
# so that import succeeds without the dependency (the adapter itself uses only
# the maintained alpaca-py, which IS installed).
if "alpaca_trade_api" not in sys.modules:
    _stub = types.ModuleType("alpaca_trade_api")
    _rest = types.ModuleType("alpaca_trade_api.rest")
    _rest.REST = object
    _stub.rest = _rest
    sys.modules["alpaca_trade_api"] = _stub
    sys.modules["alpaca_trade_api.rest"] = _rest

from reversion_bot.alpaca_py_client import (
    AlpacaPyClient,
    _build_order_request,
    _BarsResult,
)
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus


# --- order-request translation ----------------------------------------------

def _bracket_kwargs(side, type_="market", limit_price=None):
    kw = {
        "symbol": "SPY",
        "qty": 10,
        "side": side,
        "time_in_force": "day",
        "order_class": "bracket",
        "take_profit": {"limit_price": 104.0},
        "stop_loss": {"stop_price": 98.0},
        "type": type_,
    }
    if limit_price is not None:
        kw["limit_price"] = limit_price
    return kw


def test_long_market_bracket_translation():
    req = _build_order_request(_bracket_kwargs("buy"))
    assert isinstance(req, MarketOrderRequest)
    assert req.symbol == "SPY"
    assert req.qty == 10
    assert req.side == OrderSide.BUY
    assert req.time_in_force == TimeInForce.DAY
    assert req.order_class == OrderClass.BRACKET
    assert req.take_profit.limit_price == 104.0
    assert req.stop_loss.stop_price == 98.0


def test_short_market_bracket_translation():
    req = _build_order_request(_bracket_kwargs("sell"))
    assert isinstance(req, MarketOrderRequest)
    assert req.side == OrderSide.SELL
    assert req.order_class == OrderClass.BRACKET
    assert req.take_profit.limit_price == 104.0
    assert req.stop_loss.stop_price == 98.0


def test_limit_entry_bracket_carries_limit_price():
    req = _build_order_request(_bracket_kwargs("buy", type_="limit", limit_price=100.08))
    assert isinstance(req, LimitOrderRequest)
    assert req.limit_price == 100.08
    assert req.order_class == OrderClass.BRACKET


def test_plain_market_exit_has_no_bracket_legs():
    # EOD liquidation shape: market order, no order_class / take_profit / stop_loss.
    req = _build_order_request(
        {"symbol": "SPY", "qty": 5, "side": "sell", "type": "market", "time_in_force": "day"}
    )
    assert isinstance(req, MarketOrderRequest)
    assert req.order_class is None
    assert req.take_profit is None
    assert req.stop_loss is None


def test_stop_loss_carries_optional_limit_price():
    kw = _bracket_kwargs("buy")
    kw["stop_loss"] = {"stop_price": 98.0, "limit_price": 97.5}
    req = _build_order_request(kw)
    assert req.stop_loss.stop_price == 98.0
    assert req.stop_loss.limit_price == 97.5


# --- method routing over a fake alpaca-py client -----------------------------

class _FakeTradingClient:
    def __init__(self):
        self.calls = []

    def get_account(self):
        self.calls.append(("get_account",))
        return "ACCOUNT"

    def get_clock(self):
        self.calls.append(("get_clock",))
        return "CLOCK"

    def get_all_positions(self):
        self.calls.append(("get_all_positions",))
        return ["POS"]

    def get_orders(self, filter=None):
        self.calls.append(("get_orders", filter))
        return ["ORDER"]

    def submit_order(self, order_data):
        self.calls.append(("submit_order", order_data))
        return "SUBMITTED"

    def cancel_orders(self):
        self.calls.append(("cancel_orders",))
        return "CANCELLED"

    def replace_order_by_id(self, order_id, order_data):
        self.calls.append(("replace_order_by_id", order_id, order_data))
        return "REPLACED"

    def get_all_assets(self, filter=None):
        self.calls.append(("get_all_assets", filter))
        return ["ASSET"]


class _FakeTrade:
    price = 123.45


class _FakeDataClient:
    def __init__(self):
        self.calls = []

    def get_stock_latest_trade(self, request_params):
        self.calls.append(("latest_trade", request_params.symbol_or_symbols))
        # alpaca-py returns {symbol: Trade}
        return {request_params.symbol_or_symbols: _FakeTrade()}

    def get_stock_bars(self, request_params):
        self.calls.append(("bars", request_params.symbol_or_symbols))

        class _BarSet:
            df = "DATAFRAME"

        return _BarSet()


def make_adapter():
    return AlpacaPyClient(
        "k", "s", "https://paper-api.alpaca.markets",
        trading_client=_FakeTradingClient(),
        data_client=_FakeDataClient(),
    )


def test_passthrough_account_clock_positions():
    a = make_adapter()
    assert a.get_account() == "ACCOUNT"
    assert a.get_clock() == "CLOCK"
    assert a.list_positions() == ["POS"]


def test_list_orders_builds_open_filter():
    a = make_adapter()
    a.list_orders(status="open", limit=500)
    name, filt = a._trading.calls[-1]
    assert name == "get_orders"
    assert filt.status == QueryOrderStatus.OPEN
    assert filt.limit == 500


def test_submit_order_forwards_built_request():
    a = make_adapter()
    a.submit_order(**_bracket_kwargs("buy"))
    name, order_data = a._trading.calls[-1]
    assert name == "submit_order"
    assert isinstance(order_data, MarketOrderRequest)
    assert order_data.side == OrderSide.BUY


def test_cancel_all_orders_maps_to_cancel_orders():
    a = make_adapter()
    assert a.cancel_all_orders() == "CANCELLED"
    assert a._trading.calls[-1] == ("cancel_orders",)


def test_replace_order_maps_to_replace_by_id():
    a = make_adapter()
    a.replace_order("oid-1", 99.5)
    name, oid, order_data = a._trading.calls[-1]
    assert name == "replace_order_by_id"
    assert oid == "oid-1"
    assert order_data.limit_price == 99.5


def test_list_assets_builds_status_and_class_filter():
    a = make_adapter()
    a.list_assets(status="active", asset_class="us_equity")
    name, filt = a._trading.calls[-1]
    assert name == "get_all_assets"
    assert filt.status == "active"        # AssetStatus subclasses str
    assert filt.asset_class == "us_equity"


def test_get_latest_trade_unwraps_symbol_dict():
    a = make_adapter()
    trade = a.get_latest_trade("SPY")
    assert trade.price == 123.45


def test_get_bars_exposes_df_accessor():
    a = make_adapter()
    result = a.get_bars("SPY", "1Day", limit=5)
    assert isinstance(result, _BarsResult)
    assert result.df == "DATAFRAME"


# --- executor wiring (default off) -------------------------------------------

def test_executor_uses_adapter_only_when_opted_in():
    from reversion_bot.execution import AlpacaExecutor
    from reversion_bot.config import ExecutionConfig

    cfg = ExecutionConfig(base_url="https://paper-api.alpaca.markets", use_alpaca_py=True)
    ex = AlpacaExecutor("k", "s", cfg)
    assert isinstance(ex.client, AlpacaPyClient)


def test_executor_default_is_legacy_rest(monkeypatch):
    # Default off: the executor must build the legacy REST client, untouched.
    import reversion_bot.execution as execmod
    from reversion_bot.config import ExecutionConfig

    built = {}

    def fake_rest(key, secret, base_url):
        built["args"] = (key, secret, base_url)
        return object()

    monkeypatch.setattr(execmod, "REST", fake_rest)
    cfg = ExecutionConfig(base_url="https://paper-api.alpaca.markets")  # use_alpaca_py defaults False
    ex = execmod.AlpacaExecutor("k", "s", cfg)
    assert built["args"] == ("k", "s", "https://paper-api.alpaca.markets")
    assert not isinstance(ex.client, AlpacaPyClient)


# --- replace_order stop_price (audit H7) --------------------------------------
# The old signature accepted ONLY limit_price, so the trailing stop's
# replace_order(order_id=..., stop_price=...) raised TypeError on every trail
# update under USE_ALPACA_PY — swallowed upstream, trailing stop silently dead.

def test_replace_order_accepts_stop_price():
    a = make_adapter()
    a.replace_order(order_id="oid-2", stop_price=88.25)
    name, oid, order_data = a._trading.calls[-1]
    assert name == "replace_order_by_id"
    assert oid == "oid-2"
    assert order_data.stop_price == 88.25
    assert order_data.limit_price is None


def test_replace_order_requires_a_price():
    import pytest
    a = make_adapter()
    with pytest.raises(ValueError):
        a.replace_order("oid-3")


def test_executor_replace_stop_order_works_through_alpaca_py():
    # The exact live call path that was dead: AlpacaExecutor.replace_stop_order
    # -> client.replace_order(order_id=..., stop_price=...).
    from reversion_bot.execution import AlpacaExecutor
    ex = AlpacaExecutor.__new__(AlpacaExecutor)
    ex.client = make_adapter()
    ex.replace_stop_order("oid-4", 101.239)
    name, oid, order_data = ex.client._trading.calls[-1]
    assert name == "replace_order_by_id"
    assert oid == "oid-4"
    assert order_data.stop_price == 101.24     # rounded to cents
