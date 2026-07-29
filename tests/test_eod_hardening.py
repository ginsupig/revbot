"""EOD-path hardening (audit M7/M8): env parsing can't crash the loop,
half-day close times survive clock blips, EOD cancel is scoped to bot orders."""
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if "alpaca_trade_api" not in sys.modules:
    _stub = types.ModuleType("alpaca_trade_api")
    _rest = types.ModuleType("alpaca_trade_api.rest")
    _rest.REST = object
    _stub.rest = _rest
    sys.modules["alpaca_trade_api"] = _stub
    sys.modules["alpaca_trade_api.rest"] = _rest

import main
from reversion_bot.execution import AlpacaExecutor


def test_minutes_env_malformed_uses_default(monkeypatch):
    monkeypatch.setenv("EOD_LIQUIDATION_MINUTES", "15 min")
    assert main._minutes_env("EOD_LIQUIDATION_MINUTES", 10) == 10
    monkeypatch.setenv("EOD_LIQUIDATION_MINUTES", "12.5")
    assert main._minutes_env("EOD_LIQUIDATION_MINUTES", 10) == 12
    monkeypatch.setenv("EOD_LIQUIDATION_MINUTES", "")
    assert main._minutes_env("EOD_LIQUIDATION_MINUTES", 10) == 10


def test_close_time_cached_across_clock_failure():
    class Clock:
        next_close = datetime.now(timezone.utc) + timedelta(hours=2)

    class GoodClient:
        def get_clock(self):
            return Clock()

    class BadClient:
        def get_clock(self):
            raise RuntimeError("api down")

    class Ex:
        def __init__(self, client):
            self.client = client

    main._CLOSE_CT_CACHE.clear()
    good = main._get_close_ct(Ex(GoodClient()))
    # Clock API dies: the cached HALF-DAY-aware value must win over the
    # 15:00 CT guess.
    bad = main._get_close_ct(Ex(BadClient()))
    assert bad == good
    main._CLOSE_CT_CACHE.clear()


class _Order:
    def __init__(self, oid, symbol, client_order_id=""):
        self.id = oid
        self.symbol = symbol
        self.client_order_id = client_order_id


class _Client:
    def __init__(self, orders):
        self._orders = orders
        self.cancelled = []

    def list_orders(self, status=None, limit=None):
        return self._orders

    def cancel_order(self, oid):
        self.cancelled.append(oid)


def test_cancel_bot_orders_scoped_to_prefix_and_positions():
    orders = [
        _Order("1", "AAPL", "revbot-abc123"),      # bot entry -> cancel
        _Order("2", "TSLA", "manual-position"),    # manual, no position -> keep
        _Order("3", "NVDA", ""),                   # untagged leg of held pos -> cancel
        _Order("4", "MSFT", "other-bot-1"),        # other strategy -> keep
    ]
    ex = AlpacaExecutor.__new__(AlpacaExecutor)
    ex.client = _Client(orders)
    n = ex.cancel_bot_orders(position_symbols=["NVDA"])
    assert n == 2
    assert ex.client.cancelled == ["1", "3"]


def test_orders_carry_the_revbot_client_order_id():
    coid = AlpacaExecutor._make_client_order_id()
    assert coid.startswith(AlpacaExecutor.CLIENT_ORDER_PREFIX)
    assert len(coid) > len(AlpacaExecutor.CLIENT_ORDER_PREFIX) + 10
